package org.apache.spark.sql.column.utils.algorithms

import nl.liacs.mijpelaar.utils.Resources
import org.apache.arrow.algorithm.search.BucketSearcher
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, SortOrder}
import org.apache.spark.sql.column.ArrowColumnarBatchRow
import org.apache.spark.sql.column.utils.{ArrowColumnarBatchRowBuilder, ArrowColumnarBatchRowConverters, ArrowColumnarBatchRowTransformers, ArrowColumnarBatchRowUtils}

import scala.collection.mutable

object ArrowColumnarBatchRowDistributors {
  /**
   * @param key ArrowColumnarBatchRow to define distribution for
   * @param rangeBounds ArrowColumnarBatchRow containing ranges on which distribution is based
   * @param sortOrders SortOrders on which distribution is based
   * @return Indices containing the distribution for the key given the rangeBounds and sortOrders
   *         TODO: do we invalidate key and rangeBounds?
   */
  def bucketDistributor(key: ArrowColumnarBatchRow, rangeBounds: ArrowColumnarBatchRow, sortOrders: Seq[SortOrder]): Array[Int] = {
    if (key.numFields < 1 || rangeBounds.numFields < 1)
      return Array.empty

    val names: Array[String] = sortOrders.map( order => order.child.asInstanceOf[AttributeReference].name ).toArray
    Resources.autoCloseTryGet(ArrowColumnarBatchRowConverters.toUnionVector(
      ArrowColumnarBatchRowTransformers.getColumns(key, names)
    )) { keyUnion =>
      val comparator = ArrowColumnarBatchRowUtils.getComparator(keyUnion, sortOrders)

      Resources.autoCloseTryGet(ArrowColumnarBatchRowConverters.toUnionVector(
        ArrowColumnarBatchRowTransformers.getColumns(rangeBounds, names)
      )) { rangeUnion => new BucketSearcher(keyUnion, rangeUnion, comparator).distribute() }
    }
  }

  /**
   * @param key ArrowColumnarBatchRow to distribute
   * @param partitionIds Array containing which row corresponds to which partition
   * @return A map from partitionId to its corresponding ArrowColumnarBatchRow
   *
   * Caller should close the batches in the returned map
   * TODO: memory management
   */
  def distribute(key: ArrowColumnarBatchRow, partitionIds: Array[Int]): Map[Int, ArrowColumnarBatchRow] = {
    val distributed = mutable.Map[Int, ArrowColumnarBatchRowBuilder]()
    partitionIds.zipWithIndex foreach { case (partitionId, index) =>
      if (distributed.contains(partitionId))
        distributed(partitionId).append(key.copy(index until index+1,
          allocatorHint = "ArrowColumnarBatchRowDistributors::distribute"))
      else
        distributed(partitionId) = new ArrowColumnarBatchRowBuilder(key.copy(index until index+1,
          allocatorHint = "ArrowColumnarBatchRowDistributors::distribute::first"))
    }
    distributed.map ( items => (items._1, items._2.build()) ).toMap
  }
}
