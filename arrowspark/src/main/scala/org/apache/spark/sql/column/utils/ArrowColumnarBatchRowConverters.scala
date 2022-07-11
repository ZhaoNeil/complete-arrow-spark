package org.apache.spark.sql.column.utils

import nl.liacs.mijpelaar.utils.Resources
import org.apache.arrow.memory.{ArrowBuf, BufferAllocator}
import org.apache.arrow.vector.complex.UnionVector
import org.apache.arrow.vector.compression.{CompressionUtil, NoCompressionCodec}
import org.apache.arrow.vector.ipc.message.{ArrowFieldNode, ArrowRecordBatch}
import org.apache.arrow.vector.types.pojo.ArrowType.Struct
import org.apache.arrow.vector.types.pojo.FieldType
import org.apache.arrow.vector.{FieldVector, TypeLayout, VectorSchemaRoot}
import org.apache.spark.sql.column.{ArrowColumnarBatchRow, createAllocator}
import org.apache.spark.sql.vectorized.ArrowColumnVector

import scala.collection.convert.ImplicitConversions.`collection AsScalaIterable`
import java.util

/** Methods that convert an ArrowColumnarBatchRows to another type, and taking care of closing of the input */
object ArrowColumnarBatchRowConverters {
  /**
   * copied from org.apache.arrow.vector.VectorUnloader
   * Caller should close the ArrowRecordBatch
   *
   * @param batch   batch to convert and close
   * @param numCols (optional) number of columns to convert
   * @param numRows (optional) number of rows to convert
   * @return The generated ArrowRecordBatch together with the number of rows converted
   */
  def toArrowRecordBatch(batch: ArrowColumnarBatchRow, numCols: Int, numRows: Option[Int] = None): (ArrowRecordBatch, Int) = {
    Resources.autoCloseTryGet(batch) { batch =>
      val nodes = new util.ArrayList[ArrowFieldNode]
      val buffers = new util.ArrayList[ArrowBuf]
      val codec = NoCompressionCodec.INSTANCE

      val rowCount = numRows.getOrElse(batch.numRows)

      /** copied from org.apache.arrow.vector.VectorUnloader::appendNodes(...) */
      def appendNodes(vector: FieldVector, nodes: util.List[ArrowFieldNode], buffers: util.List[ArrowBuf]): Unit = {
        nodes.add(new ArrowFieldNode(rowCount, vector.getNullCount))
        val fieldBuffers = vector.getFieldBuffers
        val expectedBufferCount = TypeLayout.getTypeBufferCount(vector.getField.getType)
        if (fieldBuffers.size != expectedBufferCount)
          throw new IllegalArgumentException(String.format("wrong number of buffers for field %s in vector %s. found: %s", vector.getField, vector.getClass.getSimpleName, fieldBuffers))
        for (buf <- fieldBuffers)
          buffers.add(codec.compress(vector.getAllocator, buf))
        for (child <- vector.getChildrenFromFields)
          appendNodes(child, nodes, buffers)
      }

      batch.columns.slice(0, numCols) foreach (column => appendNodes(column.getValueVector.asInstanceOf[FieldVector], nodes, buffers))
      (new ArrowRecordBatch(rowCount, nodes, buffers, CompressionUtil.createBodyCompression(codec), true), rowCount)
    }
  }

  /** Creates a VectorSchemaRoot from the provided batch and closes it
   * Returns the root and the number of rows transferred
   * Caller should close the root
   */
  def toRoot(batch: ArrowColumnarBatchRow, numCols: Option[Int] = None, numRows: Option[Int] = None): (VectorSchemaRoot, Int) = {
    Resources.autoCloseTryGet(batch) { batch =>
      val rowCount = numRows.getOrElse(batch.numRows)
      val columns = batch.columns.slice(0, numCols.getOrElse(batch.numFields))
      (VectorSchemaRoot.of(columns.map(column => {
        val vector = column.getValueVector
        val tp = vector.getTransferPair(vector.getAllocator.newChildAllocator("ArrowColumnarBatchRowConverters::toRoot", 0, org.apache.spark.sql.column.perAllocatorSize))
        tp.splitAndTransfer(0, rowCount)
        tp.getTo.asInstanceOf[FieldVector]
      }).toSeq: _*), rowCount)
    }
  }

  /**
   * Splits a single batch into two
   *
   * @param batch    ArrowColumnarBatchRow to split and close
   * @param rowIndex index to split on
   * @return two ArrowColumnarBatchRows split on rowIndex from batch
   *         If rowIndex > batch.numRows, returns (original-batch, empty-batch)
   *         Caller is responsible for closing the batches
   */
  def split(batch: ArrowColumnarBatchRow, rowIndex: Int): (ArrowColumnarBatchRow, ArrowColumnarBatchRow) = {
    Resources.autoCloseTryGet(batch) { batch =>
      val splitPoint = rowIndex.min(batch.numRows)
      (batch.copy(createAllocator("ArrowColumnarBatchRowConverters::split::first"), 0 until splitPoint),
        batch.copy(createAllocator("ArrowColumnarBatchRowConverters::split::second"), splitPoint until batch.numRows))
    }
  }

  /**
   * Splits the current batch on its columns into two batches
   *
   * @param batch batch to split and close
   * @param col   column index to split on, we assume col < batch.numFields
   * @return a pair of two of the batches containing the split columns from this batch
   *         Caller is responsible for closing the two returned batches
   */
  def splitColumns(batch: ArrowColumnarBatchRow, col: Int): (ArrowColumnarBatchRow, ArrowColumnarBatchRow) = {
    Resources.autoCloseTryGet(batch) { batch =>
      val firstBatch = {
        val firstAllocator = createAllocator("ArrowColumnarBatchRowConverters::splitColumns::first")
        // FIXME: close if allocation fails while performing the map
        val columns = batch.columns.slice(0, col).map { column =>
          val vector = column.getValueVector
          val tp = vector.getTransferPair(createAllocator(firstAllocator, vector.getName))
          tp.transfer()
          new ArrowColumnVector(tp.getTo)
        }
        new ArrowColumnarBatchRow(firstAllocator, columns, batch.numRows)
      }
      Resources.closeTraversableOnFailGet(firstBatch) { firstBatch =>
        val secondAllocator = createAllocator("ArrowColumnarBatchRowConverters::splitColumns::second")
        // FIXME: close if allocation fails while performing the map
        val columns = batch.columns.slice(col, batch.numFields).map { column =>
          val vector = column.getValueVector
          val tp = vector.getTransferPair(createAllocator(secondAllocator, vector.getName))
          tp.transfer()
          new ArrowColumnVector(tp.getTo)
        }
        (firstBatch, new ArrowColumnarBatchRow(secondAllocator, columns, batch.numRows))
      }
    }
  }

  /**
   * Creates an UnionVector from the provided batch
   *
   * @param batch batch to convert and close
   * @return a fresh UnionVector
   *         Caller is responsible for closing the UnionVector
   */
  def toUnionVector(batch: ArrowColumnarBatchRow): UnionVector = {
    Resources.autoCloseTryGet(batch) { batch =>
      Resources.closeTraversableOnFailGet(new UnionVector("Combiner",
        createAllocator("ArrowColumnarBatchRowConverters::toUnionVector::union"), FieldType.nullable(Struct.INSTANCE), null)) { union =>
        batch.columns foreach { column =>
          val vector = column.getValueVector
          val tp = vector.getTransferPair(createAllocator(union.getAllocator, vector.getName))
          tp.transfer()
          union.addVector(tp.getTo.asInstanceOf[FieldVector])
        }
        union.setValueCount(batch.numRows)

        union
      }
    }
  }

  /**
   * Creates an array of fresh ArrowColumnVectors with the same type as the given batch
   *
   * @param parentAllocator [[BufferAllocator]] to use as parent-Allocator for the columns
   * @param batch           ArrowColumnarBatchRow to create array from and close
   * @return An array of fresh ArrowColumnVectors from the provided batch
   *         Caller is responsible for closing the vectors in the array
   */
  def makeFresh(parentAllocator: BufferAllocator, batch: ArrowColumnarBatchRow): Array[ArrowColumnVector] = {
    Resources.closeTraversableOnFailGet(batch) { batch =>
      Array.tabulate[ArrowColumnVector](batch.numFields) { i =>
        val vector = batch.columns(i).getValueVector
        val tp = vector.getTransferPair(createAllocator(parentAllocator, vector.getName))
        // we 'copy' the content of the first batch ...
        tp.transfer()
        // ... and re-use the ValueVector so we do not have to determine vector types :)
        val new_vec = tp.getTo
        new_vec.clear()
        new_vec.allocateNew()
        new ArrowColumnVector(new_vec)
      }
    }
  }
}
