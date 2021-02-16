package feature.transformer

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SparkSession

/**
 * @author Shi Lei
 * @create 2021-02-13
 */
object StringIndexerL {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]")
      .appName("ml").getOrCreate()
    spark.sparkContext.setLogLevel("warn")

    val df = spark.createDataFrame(
      Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
    ).toDF("id", "category")

    df.show(false)
    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")

    val indexed = indexer.fit(df).transform(df)
    indexed.show(false)

  }
}
