package org.apache.spark.sql.execution

import org.apache.arrow.memory.RootAllocator
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{And, Attribute, AttributeReference, BoundReference, Expression, PlanExpression, Predicate}
import org.apache.spark.sql.catalyst.{InternalRow, TableIdentifier}
import org.apache.spark.sql.column.ArrowColumnarBatchRow
import org.apache.spark.sql.execution.datasources._
import org.apache.spark.sql.sources.{BaseRelation, Filter}
import org.apache.spark.sql.types.{ArrayType, StructType}

import scala.collection.mutable


trait ArrowFileFormat extends FileFormat {
  /** Returns a function that can be used to read a single file in as an Iterator of Array[ValueVector]
   * Caller should close batches in Iterator */
  def buildArrowReaderWithPartitionValues(sparkSession: SparkSession,
                                          dataSchema: StructType,
                                          partitionSchema: StructType,
                                          requiredSchema: StructType,
                                          filters: Seq[Filter],
                                          options: Map[String, String],
                                          hadoopConf: Configuration) : (PartitionedFile, RootAllocator) => Iterator[ArrowColumnarBatchRow]

  override def supportBatch(sparkSession: SparkSession, dataSchema: StructType): Boolean = true

  /** Note: children should use this to change their schema to one appropriate for spark */
  def inferSchema(schema: Option[StructType]): Option[StructType] = {
    schema.map( s =>  s.copy( fields = s.fields.map(field => field.copy(dataType = ArrayType.apply(field.dataType)))))
  }
}

/** Caller should close whatever is gathered from this plan */
case class ArrowScanExec(fs: FileSourceScanExec) extends DataSourceScanExec with Logging {
  // copied from org/apache/spark/sql/execution/DataSourceScanExec.scala
  @transient
  private lazy val pushedDownFilters = {
    val supportNestedPredicatePushdown = DataSourceUtils.supportNestedPredicatePushdown(fs.relation)
    fs.dataFilters.flatMap(DataSourceStrategy.translateFilter(_, supportNestedPredicatePushdown))
  }

  // copied and edited from org/apache/spark/sql/execution/DataSourceScanExec.scala
  // caller should close batches in RDD
  private def createFileScanArrowRDD(readFunc: (PartitionedFile, RootAllocator) => Iterator[ArrowColumnarBatchRow],
                                     selectedPartitions: Array[PartitionDirectory],
                                     fsRelation: HadoopFsRelation): FileScanArrowRDD = {
    val openCostInBytes = fsRelation.sparkSession.sessionState.conf.filesOpenCostInBytes
    val maxSplitBytes = FilePartition.maxSplitBytes(fsRelation.sparkSession, selectedPartitions)
    logInfo(s"Planning scan with bin packing, max size: $maxSplitBytes bytes, " +
      s"open cost is considered as scanning $openCostInBytes bytes.")

    // Filter files with bucket pruning if possible
    val bucketingEnabled = fsRelation.sparkSession.sessionState.conf.bucketingEnabled
    val shouldProcess: Path => Boolean = fs.optionalBucketSet match {
      case Some(bucketSet) if bucketingEnabled =>
        // Do not prune the file if bucket file name is invalid
        filePath => BucketingUtils.getBucketId(filePath.getName).forall(bucketSet.get)
      case _ =>
        _ => true
    }

    val splitFiles = selectedPartitions.flatMap { partition =>
      partition.files.flatMap { file =>
        // getPath() is very expensive so we only want to call it once in this block:
        val filePath = file.getPath

        if (shouldProcess(filePath)) {
          val isSplitable = fs.relation.fileFormat.isSplitable(
            fs.relation.sparkSession, fs.relation.options, filePath)
          PartitionedFileUtil.splitFiles(
            sparkSession = fs.relation.sparkSession,
            file = file,
            filePath = filePath,
            isSplitable = isSplitable,
            maxSplitBytes = maxSplitBytes,
            partitionValues = partition.values
          )
        } else {
          Seq.empty
        }
      }
    }.sortBy(_.length)(implicitly[Ordering[Long]].reverse)

    val partitions =
      ArrowFilePartition.getFilePartitions(fs.relation.sparkSession, splitFiles, maxSplitBytes)

    new FileScanArrowRDD(fsRelation.sparkSession, readFunc, partitions)
  }

  // copied and edited from org/apache/spark/sql/execution/DataSourceScanExec.scala
  // caller should close batch in RDD
  private def createBucketFileScanArrowRDD(readFunc: (PartitionedFile, RootAllocator) => Iterator[ArrowColumnarBatchRow],
                                           numBuckets: Int,
                                           selectedPartitions: Array[PartitionDirectory]): FileScanArrowRDD  = {
    logInfo(s"Planning with $numBuckets buckets")
    val filesGroupedToBuckets =
      selectedPartitions.flatMap { p =>
        p.files.map { f =>
          PartitionedFileUtil.getPartitionedFile(f, f.getPath, p.values)
        }
      }.groupBy { f =>
        BucketingUtils
          .getBucketId(new Path(f.filePath.toString()).getName)
          .getOrElse(throw new IllegalStateException(s"Invalid bucket file ${f.filePath}"))
      }

    val prunedFilesGroupedToBuckets = if (fs.optionalBucketSet.isDefined) {
      val bucketSet = fs.optionalBucketSet.get
      filesGroupedToBuckets.filter {
        f => bucketSet.get(f._1)
      }
    } else {
      filesGroupedToBuckets
    }

    val filePartitions = fs.optionalNumCoalescedBuckets.map { numCoalescedBuckets =>
      logInfo(s"Coalescing to $numCoalescedBuckets buckets")
      val coalescedBuckets = prunedFilesGroupedToBuckets.groupBy(_._1 % numCoalescedBuckets)
      // Note: IntelliJ marks the asInstance as redundant, but it is required, please keep it there
      Seq.tabulate(numCoalescedBuckets) { bucketId =>
        val partitionedFiles = coalescedBuckets.get(bucketId).map {
          _.values.flatten.toArray
        }.getOrElse(Array.empty).asInstanceOf[Array[org.apache.spark.sql.execution.datasources.PartitionedFile]]
        ArrowFilePartition(FilePartition(bucketId, partitionedFiles))
      }
    }.getOrElse {
      Seq.tabulate(numBuckets) { bucketId =>
        ArrowFilePartition(FilePartition(bucketId, prunedFilesGroupedToBuckets.getOrElse(bucketId, Array.empty)))
      }
    }

    new FileScanArrowRDD(fs.relation.sparkSession, readFunc, filePartitions)
  }

  // copied from org/apache/spark/sql/execution/DataSourceScanExec.scala
  private def isDynamicPruningFilter(e: Expression): Boolean =
    e.find(_.isInstanceOf[PlanExpression[_]]).isDefined

  // copied from org/apache/spark/sql/execution/DataSourceScanExec.scala
  private lazy val driverMetrics: mutable.HashMap[String, Long] = mutable.HashMap.empty

  // copied from org/apache/spark/sql/execution/DataSourceScanExec.scala
  /** Helper for computing total number and size of files in selected partitions. */
  private def setFilesNumAndSizeMetric(
                                        partitions: Seq[PartitionDirectory],
                                        static: Boolean): Unit = {
    val filesNum = partitions.map(_.files.size.toLong).sum
    val filesSize = partitions.map(_.files.map(_.getLen).sum).sum
    if (!static || !fs.partitionFilters.exists(isDynamicPruningFilter)) {
      driverMetrics("numFiles") = filesNum
      driverMetrics("filesSize") = filesSize
    } else {
      driverMetrics("staticFilesNum") = filesNum
      driverMetrics("staticFilesSize") = filesSize
    }
  }

  // copied and edited from org/apache/spark/sql/execution/DataSourceScanExec.scala
  //
  // We can only determine the actual partitions at runtime when a dynamic partition filter is
  // present. This is because such a filter relies on information that is only available at run
  // time (for instance the keys used in the other side of a join).
  @transient private lazy val dynamicallySelectedPartitions: Array[PartitionDirectory] = {
    val dynamicPartitionFilters = fs.partitionFilters.filter(isDynamicPruningFilter)

    if (dynamicPartitionFilters.nonEmpty) {
      val startTime = System.nanoTime()
      // call the file index for the files matching all filters except dynamic partition filters
      val predicate = dynamicPartitionFilters.reduce(And)
      val partitionColumns = fs.relation.partitionSchema
      val boundPredicate = Predicate.create(predicate.transform {
        case a: AttributeReference =>
          val index = partitionColumns.indexWhere(a.name == _.name)
          BoundReference(index, partitionColumns(index).dataType, nullable = true)
      }, Nil)
      val ret = fs.selectedPartitions.filter(p => boundPredicate.eval(p.values))
      setFilesNumAndSizeMetric(ret, static = false)
      val timeTakenMs = (System.nanoTime() - startTime) / 1000 / 1000
      driverMetrics("pruningTime") = timeTakenMs
      ret
    } else {
      fs.selectedPartitions
    }
  }

  lazy val inputRDD: RDD[InternalRow] = {
    val readFunction: (PartitionedFile, RootAllocator) => Iterator[ArrowColumnarBatchRow] = fs.relation.fileFormat.asInstanceOf[ArrowFileFormat].buildArrowReaderWithPartitionValues(
      fs.relation.sparkSession, fs.relation.dataSchema, fs.relation.partitionSchema, fs.requiredSchema, pushedDownFilters,
      fs.relation.options,  fs.relation.sparkSession.sessionState.newHadoopConfWithOptions(fs.relation.options)
    )
    if (fs.bucketedScan)
      createBucketFileScanArrowRDD(readFunction, fs.relation.bucketSpec.get.numBuckets, dynamicallySelectedPartitions).asInstanceOf[RDD[InternalRow]]
    else
      createFileScanArrowRDD(readFunction, dynamicallySelectedPartitions, fs.relation).asInstanceOf[RDD[InternalRow]]
  }

  override def relation: BaseRelation = fs.relation

  override def tableIdentifier: Option[TableIdentifier] = fs.tableIdentifier

  override protected def metadata: Map[String, String] = fs.metadata

  /** Caller should close returned InternalRow */
  override def inputRDDs(): Seq[RDD[InternalRow]] = inputRDD :: Nil

  /** Caller should close returned InternalRow */
  override protected def doExecute(): RDD[InternalRow] = inputRDD

  override def output: Seq[Attribute] = fs.output
}