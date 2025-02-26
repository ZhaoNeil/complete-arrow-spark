package nl.liacs.mijpelaar.evaluation

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.column._

import java.io.FileWriter
import java.nio.file.Paths
import scala.reflect.io.Directory

object EvaluationSuite {
  /** Sort on the first two columns (which we assume are integers) of a parquet file */
  def sort(spark: SparkSession, fw: FileWriter, file: String): Unit = {
    val df = spark.read.parquet(file)
    val cols = df.columns
    assert(cols.length > 0)
    val sorted_df = if (cols.length == 1) df.sort(cols(0)) else df.sort(cols(0), cols(1))
    val vanilla_start = System.nanoTime()
    val rdd = sorted_df.queryExecution.executedPlan.execute()
    val func: Iterator[InternalRow] => Int = { iter => iter.length }
    spark.sparkContext.runJob(rdd, func).sum
    val vanilla_stop = System.nanoTime()
    fw.write("Vanilla compute: %04.3f\n".format((vanilla_stop-vanilla_start)/1e9d))
    fw.flush()

    val cdf: ColumnDataFrame =
      new ColumnDataFrameReader(spark).format("org.apache.spark.sql.execution.datasources.SimpleParquetArrowFileFormat")
        .loadDF(file)
    val cCols = cdf.columns
    assert(cCols.length > 0)
    val sorted_cdf = if (cCols.length == 1) cdf.sort(cCols(0)) else cdf.sort(cCols(0), cCols(1))
    val cas_start = System.nanoTime()
    val arrowRDD = sorted_cdf.queryExecution.executedPlan.execute()
    val arrowFunc: Iterator[InternalRow] => Int = { case iter: Iterator[ArrowColumnarBatchRow] =>
      iter.map { batch =>
        try {
          batch.numRows
        } finally {
          batch.close()
        }
      }.sum
    }
    spark.sparkContext.runJob(arrowRDD, arrowFunc).sum
    val cas_stop = System.nanoTime()
    fw.write("CAS compute: %04.3f\n".format((cas_stop-cas_start)/1e9d))
    fw.flush()
  }

  /** Sort on the first two columns (which we assume are integers) of a directory of parquet files */
  def sort(spark: SparkSession, fw: FileWriter, dir: Directory, onlyVanilla: Boolean, onlyCas: Boolean): Unit = {
    if (!onlyCas) {
      val tableName = "vanilla"
      spark.read.format("parquet").option("mergeSchema", "true").option("dbtable", tableName)
        .load(Paths.get(dir.toString()).resolve("*").toString)
        .createOrReplaceTempView(tableName)
      val df = spark.table(tableName)
      val cols = df.columns
      assert(cols.length > 0)
      val sorted_df = if (cols.length == 1) df.sort(cols(0)) else df.sort(cols(0), cols(1))
      val vanilla_start = System.nanoTime()
      val n1 = System.nanoTime()
      val rdd = sorted_df.queryExecution.executedPlan.execute()
      val n2 = System.nanoTime()
      val func: Iterator[InternalRow] => Int = { iter => iter.length }
      spark.sparkContext.runJob(rdd, func).sum
      val n3 = System.nanoTime()
      val vanilla_stop = System.nanoTime()
      val readingTimeVan = (n2 - n1) / 1e9d
      val sortingTimeVan = (n3 - n2) / 1e9d
      fw.write("Vanilla stage 1: %04.3f\n".format(readingTimeVan))
      fw.write("Vanilla stage 2: %04.3f\n".format(sortingTimeVan))
      fw.write("Vanilla total: %04.3f\n".format((vanilla_stop-vanilla_start)/1e9d))
      fw.flush()
    }

    if (!onlyVanilla) {
      val cdf: ColumnDataFrame =
        new ColumnDataFrameReader(spark).format("org.apache.spark.sql.execution.datasources.SimpleParquetArrowFileFormat")
          .loadDF(dir.path)
      val cCols = cdf.columns
      assert(cCols.length > 0)
      val sorted_cdf = if (cCols.length == 1) cdf.sort(cCols(0)) else cdf.sort(cCols(0), cCols(1))
      val cas_start = System.nanoTime()
      val t1 = System.nanoTime()
      val arrowRDD = sorted_cdf.queryExecution.executedPlan.execute()
      val t2 = System.nanoTime()
      val arrowFunc: Iterator[InternalRow] => Int = { case iter: Iterator[ArrowColumnarBatchRow] =>
        iter.map { batch =>
          try {
            batch.numRows
          } finally {
            batch.close()
          }
        }.sum
      }
      spark.sparkContext.runJob(arrowRDD, arrowFunc).sum
      val t3 = System.nanoTime()
      val cas_stop = System.nanoTime()
      val readingTime = (t2 - t1) / 1e9d
      val sortingTime = (t3 - t2) / 1e9d
      fw.write("CAS stage 1: %04.3f\n".format(readingTime))
      fw.write("CAS stage 2: %04.3f\n".format(sortingTime))
      fw.write("CAS total: %04.3f\n".format((cas_stop-cas_start)/1e9d))
      fw.flush()
    }
  }
}
