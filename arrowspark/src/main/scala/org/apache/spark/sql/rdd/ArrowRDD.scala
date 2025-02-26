package org.apache.spark.sql.rdd

import nl.liacs.mijpelaar.utils.Reporter
import org.apache.arrow.memory.RootAllocator
import org.apache.spark.internal.config.RDD_LIMIT_SCALE_UP_FACTOR
import org.apache.spark.rdd.{RDD, RDDOperationScope}
import org.apache.spark.sql.column.AllocationManager.newRoot
import org.apache.spark.sql.column.ArrowColumnarBatchRow
import org.apache.spark.sql.column.utils.{ArrowColumnarBatchRowEncoders, ArrowColumnarBatchRowUtils}
import org.apache.spark.sql.internal.ArrowConf
import org.apache.spark.sql.internal.ArrowConf.ARROWRDD_REPORT_DIRECTORY
import org.apache.spark.{ArrowPartition, Partition, SparkEnv, TaskContext}

import java.io.File
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.reflect.io.Directory

// Caller should close batches in RDD
trait ArrowRDD extends RDD[ArrowColumnarBatchRow] {
  protected val reportDirectory: String = ArrowConf.get(sparkContext, ARROWRDD_REPORT_DIRECTORY)

  // Caller should close returned batches
  // Closes the batches in the RDD
  override def collect(): Array[ArrowColumnarBatchRow] = ArrowRDD.collect(this).map(_._2)
  // Caller should close returned batches
  // Closes the batches in the RDD
  override def take(num: Int): Array[ArrowColumnarBatchRow] = ArrowRDD.take(num, this)
  // Caller should close returned batches
  // Closes the batches in the RDD
  override def toLocalIterator: Iterator[ArrowColumnarBatchRow] = ArrowRDD.toLocalIterator(this)

  override def compute(split: Partition, context: TaskContext): Iterator[ArrowColumnarBatchRow] = split match {
    case arrowPartition: ArrowPartition =>
      val executorId = Option(SparkEnv.get).getOrElse( throw new RuntimeException("ArrowRDD: cannot get SparkEnv")).executorId
      val stageId = context.stageId()
      context.addTaskCompletionListener[Unit]( _ =>
        try {
//          arrowPartition.allocator.close()
          Reporter.report(new Directory(new File(reportDirectory)), executorId = executorId, stageId = stageId.toString, partitionId=split.toString)
        } catch {
          case e: Throwable =>
            println("---------------DEBUG------------------")
            println("ArrowPartition")
            println(arrowPartition.allocator.toVerboseString)
            println("--------------------------------------")
            throw e
        }

      )
      //      println(s"compute: ${arrowPartition.asInstanceOf[ArrowFilePartition].filePartition.files.map(file => file.toString()).mkString("Array(", ", ", ")")}")
      compute(arrowPartition, context)
    case _ => throw new IllegalArgumentException(s"ArrowRDD can only accept ArrowPartitions, not ${split.getClass.getName}")
  }

  def compute(split: ArrowPartition, context: TaskContext): Iterator[ArrowColumnarBatchRow]
}

object ArrowRDD {
  /** [performance] Spark's internal mapPartitions method that skips closure cleaning
   * Inspired from: org.apache.spark.rdd.RDD::mapPartitionsInternal */
  private[spark] def mapPartitionsInternal[U: ClassTag, T: ClassTag](
      rdd: RDD[T],
      f: (RootAllocator, Iterator[T]) => Iterator[U],
      preservePartitioning: Boolean = false): RDD[U] =
    RDDOperationScope.withScope(rdd.sparkContext) {
    new ArrowMapPartitionsRDD[U, T](rdd, (_, _, root, iter) => f(root, iter), preservePartitioning)
  }

  /** Returns a new RDD by applying a function to all elements of this RDD
   * Inspired from: org.apache.spark.rdd.RDD::map */
  def map[U: ClassTag, T: ClassTag](rdd: RDD[T], f: (RootAllocator, T) => U): RDD[U] =
    RDDOperationScope.withScope(rdd.sparkContext) {
      val cleanF = rdd.sparkContext.clean(f)
      new ArrowMapPartitionsRDD[U, T](rdd, (_, _, rootAllocator, iter) =>
        iter.map(item => cleanF(rootAllocator, item)) )
    }

  /** Return a new RDD by applying a function to each partition of this RDD, while tracking the index of the original partition.
  preservesPartitioning indicates whether the input function preserves the partitioner,
  which should be false unless this is a pair RDD and the input function doesn't modify the keys.
   Inspired from: org.apache.spark.rdd.RDD:mapPartitionsWithIndex*/
  def mapPartitionsWithIndex[U: ClassTag, T: ClassTag](
      rdd: RDD[T], f: (Int, RootAllocator, Iterator[T]) => Iterator[U], preservePartitioning: Boolean = false): RDD[U] =
    RDDOperationScope.withScope(rdd.sparkContext) {
      new ArrowMapPartitionsRDD(rdd,
        (_, index, rootAllocator, iter: Iterator[T]) => f(index, rootAllocator, iter), preservePartitioning)
    }

  /** Returns a local iterator for each partition
   * Caller should cose batches in the returned iterator */
  def toLocalIterator(rdd: RDD[ArrowColumnarBatchRow], rootAllocator: Option[RootAllocator] = None): Iterator[ArrowColumnarBatchRow] = {
    val childRDD = rdd.mapPartitionsInternal( res => ArrowColumnarBatchRowEncoders.encode(res))
    childRDD.toLocalIterator.flatMap( result =>
      ArrowColumnarBatchRowEncoders.decode(rootAllocator.getOrElse(newRoot()), result).asInstanceOf[Iterator[ArrowColumnarBatchRow]]
    )
  }

  /**
   * Collect utility for rdds that contain ArrowColumnarBatchRows. Users can pass optional functions to process data
   * if the rdd has more complex data than only ArrowColumnarBatchRows
   * @param rdd rdd with the batches, which is also closed
   * @param rootAllocator (optional) [[RootAllocator]] to allocate batches with, otherwise, each batch gets its own allocator
   * @param extraEncoder (optional) split item into encoded custom-data and a batch
   * @param extraDecoder (optional) decode an array of bytes to custom-data and a batch to a single instance
   * @param extraTaker (optional) split the item from the iterator into (customData, batch)
   * @return array of custom-data and batches, and the [[RootAllocator]] used to allocate the batches
   *         Caller should close the batches in the array, and their RootAllocators
   */
  def collect[T: ClassTag](rdd: RDD[T],
                           rootAllocator: Option[RootAllocator] = None,
                           extraEncoder: Any => (Array[Byte], ArrowColumnarBatchRow) = batch => (Array.emptyByteArray, batch.asInstanceOf[ArrowColumnarBatchRow]),
                           extraDecoder: (Array[Byte], ArrowColumnarBatchRow) => Any = (_, batch) => batch,
                           extraTaker: Any => (Any, ArrowColumnarBatchRow) = batch => (None, batch.asInstanceOf[ArrowColumnarBatchRow]))
                          (implicit ct: ClassTag[T]): (Array[(Any, ArrowColumnarBatchRow)]) = {
    val childRDD = rdd.mapPartitionsInternal { res => ArrowColumnarBatchRowEncoders.encode(res, extraEncoder = extraEncoder)}
    val res = rdd.sparkContext.runJob(childRDD, (it: Iterator[Array[Byte]]) => {
      if (!it.hasNext) Array.emptyByteArray else it.next()
    })
    // FIXME: For now, we assume we do not return too early when building the buf
    val buf = new ArrayBuffer[(Any, ArrowColumnarBatchRow)]
    res.foreach(result => {
      val decoded = ArrowColumnarBatchRowEncoders.decode(rootAllocator.getOrElse(newRoot()), result, extraDecoder = extraDecoder)
      buf ++= decoded.map( item => extraTaker(item) )
    })
    buf.toArray
  }

  /** Note: copied and adapted from RDD.scala
   * batches in RDD are consumed (closed)
   * Caller should close returned batches */
  def take(num: Int, rdd: RDD[ArrowColumnarBatchRow], rootAllocator: Option[RootAllocator] = None): Array[ArrowColumnarBatchRow] = {
    if (num == 0) new Array[ArrowColumnarBatchRow](0)

    val scaleUpFactor = Math.max(rdd.conf.get(RDD_LIMIT_SCALE_UP_FACTOR), 2)
    // FIXME: For now, we assume we do not return too early
    val buf = new ArrayBuffer[ArrowColumnarBatchRow]
    val totalParts = rdd.partitions.length
    var partsScanned = 0
    val childRDD = rdd.mapPartitionsInternal { res => ArrowColumnarBatchRowEncoders.encode(res, numRows = Option(num)) }

    while (buf.size < num && partsScanned < totalParts) {
      // The number of partitions to try in this iteration. It is ok for this number to be
      // greater than totalParts because we actually cap it at totalParts in runJob.
      var numPartsToTry = 1L
      val left = num - buf.size
      if (partsScanned > 0) {
        // If we didn't find any rows after the previous iteration, quadruple and retry.
        // Otherwise, interpolate the number of partitions we need to try, but overestimate
        // it by 50%. We also cap the estimation in the end.
        if (buf.isEmpty) {
          numPartsToTry = partsScanned * scaleUpFactor
        } else {
          // As left > 0, numPartsToTry is always >= 1
          numPartsToTry = Math.ceil(1.5 * left * partsScanned / buf.size).toInt
          numPartsToTry = Math.min(numPartsToTry, partsScanned * scaleUpFactor)
        }
      }

      val p = partsScanned.until(math.min(partsScanned + numPartsToTry, totalParts).toInt)
      val res = childRDD.sparkContext.runJob(childRDD, (it: Iterator[Array[Byte]]) => {
        if (!it.hasNext) Array.emptyByteArray else it.next()
      }, p)

      res.foreach(result => {
        // NOTE: we require the 'take', because we do not want more than num numRows
        val root = rootAllocator.getOrElse(newRoot())
        val decoded = ArrowColumnarBatchRowUtils.take(root, ArrowColumnarBatchRowEncoders.decode(root, result), numRows = Option(num))
        buf += ArrowColumnarBatchRow.create(decoded._3, decoded._2)
      })

      partsScanned += p.size
    }

    buf.toArray
  }

}
