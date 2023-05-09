package org.apache.arrow.util.vector.read

import nl.liacs.mijpelaar.utils.Resources
import org.apache.arrow.dataset.file.{FileFormat, FileSystemDatasetFactory}
import org.apache.arrow.dataset.jni.NativeMemoryPool
import org.apache.arrow.dataset.scanner.{ScanOptions, ScanTask, Scanner}
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.{VectorLoader, VectorSchemaRoot, VectorUnloader}
import org.apache.spark.TaskContext
import org.apache.spark.sql.column
import org.apache.spark.sql.column.AllocationManager.createAllocator
import org.apache.spark.sql.column.ArrowColumnarBatchRow
import org.apache.spark.sql.execution.datasources.PartitionedFile
import org.apache.spark.sql.vectorized.ArrowColumnVector

import java.util
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable.ArrayBuffer

/** inspired from: https://arrow.apache.org/cookbook/java/dataset.html#query-parquet-file */
class ArrowParquetReaderIterator(protected val file: PartitionedFile, protected val rootAllocator: RootAllocator) extends Iterator[ArrowColumnarBatchRow] with AutoCloseable {
  if (file.length > column.AllocationManager.perAllocatorSize)
    throw new RuntimeException("[ArrowParquetReaderIterator] Partition is too large")

  private val closeables: ArrayBuffer[AutoCloseable] = ArrayBuffer.empty
  def close(): Unit = {
    closeables.foreach(_.close())
  }
  private def scheduleClose[T <: AutoCloseable](closeables: T*): Unit = {
    closeables.foreach(this.closeables.append(_))
    Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => {
      closeables.foreach(_.close())
    }))
  }

  val scanner: Scanner = {
    Resources.autoCloseTryGet(new FileSystemDatasetFactory(rootAllocator, NativeMemoryPool.getDefault, FileFormat.PARQUET, file.filePath.toString())) { factory =>
      val dataset = factory.finish()
      // TODO: make configurable?
      val scanner = dataset.newScan(new ScanOptions(Integer.MAX_VALUE))
      scheduleClose(dataset, scanner)
      scanner
    }
  }

  val root: VectorSchemaRoot = {
    val root = VectorSchemaRoot.create(scanner.schema(), rootAllocator)
    scheduleClose(root)
    root
  }

  val arrowReader = scanner.scanBatches()
  val unloader: VectorUnloader = new VectorUnloader(arrowReader.getVectorSchemaRoot)

  override def hasNext: Boolean = arrowReader.loadNextBatch()

  override def next(): ArrowColumnarBatchRow = {
    Resources.autoCloseTryGet(unloader.getRecordBatch) { recordBatch =>
      val loader: VectorLoader = new VectorLoader(root)
      loader.load(recordBatch)
      /** Transfer ownership to new batch */
      Resources.autoCloseTraversableTryGet(root.getFieldVectors.asScala.toIterator) { data =>
        val allocator = createAllocator(rootAllocator, "ArrowParquetReaderIterator::transfer")
        val transferred = data.map { fieldVector =>
          val tp = fieldVector.getTransferPair(createAllocator(allocator, fieldVector.getName))
          tp.splitAndTransfer(0, fieldVector.getValueCount)
          new ArrowColumnVector(tp.getTo)
        }
        new ArrowColumnarBatchRow(allocator, transferred.toArray, root.getRowCount)
      }
    }
  }
}