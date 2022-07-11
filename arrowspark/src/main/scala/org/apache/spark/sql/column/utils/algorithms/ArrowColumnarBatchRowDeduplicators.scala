package org.apache.spark.sql.column.utils.algorithms

import nl.liacs.mijpelaar.utils.Resources
import org.apache.arrow.algorithm.deduplicate.VectorDeduplicator
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, SortOrder}
import org.apache.spark.sql.column.ArrowColumnarBatchRow
import org.apache.spark.sql.column.utils.{ArrowColumnarBatchRowConverters, ArrowColumnarBatchRowTransformers, ArrowColumnarBatchRowUtils}

object ArrowColumnarBatchRowDeduplicators {

  /**
   * @param batch batch to gather unique values from
   * @param sortOrders order to define unique-ness
   * @return a fresh batch with the unique values from the previous
   *
   * Caller is responsible for closing the new batch
   * TODO: do we invalidate batch?
   */
  def unique(batch: ArrowColumnarBatchRow, sortOrders: Seq[SortOrder]): ArrowColumnarBatchRow = {
    if (batch.numFields < 1)
      return batch

    // UnionVector representing our batch
    Resources.autoCloseTryGet(ArrowColumnarBatchRowConverters.toUnionVector(
      ArrowColumnarBatchRowTransformers.getColumns(batch,
        sortOrders.map( order => order.child.asInstanceOf[AttributeReference].name).toArray)
    )) { union =>
      val comparator = ArrowColumnarBatchRowUtils.getComparator(union, sortOrders)
      ArrowColumnarBatchRowTransformers.applyIndices(batch, VectorDeduplicator.uniqueIndices(comparator, union))
    }
  }

}
