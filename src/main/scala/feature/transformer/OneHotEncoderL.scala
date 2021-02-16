package feature.transformer

import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderEstimator, StringIndexer}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * @author Shi Lei
 * @create 2021-02-13
 */
object OneHotEncoderL {
  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession.builder().master("local[*]")
      .appName("ml").getOrCreate()
    spark.sparkContext.setLogLevel("warn")

    val df = spark.createDataFrame(Seq(
      (0.0, 1.0),
      (1.0, 0.0),
      (2.0, 1.0),
      (0.0, 2.0),
      (0.0, 1.0),
      (2.0, 0.0)
    )).toDF("categoryIndex1", "categoryIndex2")

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("categoryIndex1", "categoryIndex2"))
      .setOutputCols(Array("categoryVec1", "categoryVec2"))
    val model = encoder.fit(df)
    val encoded = model.transform(df)
    encoded.show(false)


    println("=============demo 2======================")

    val df2 = spark.createDataFrame(Seq(
      (0.0, "上海"),
      (1.0, "北京"),
      (2.0, "广州"),
      (3.0, "深证"),
      (2.0, "广州"),
      (0.0, "上海")
    )).toDF("num", "city")

    val indexer: StringIndexer = new StringIndexer().setInputCol("city").setOutputCol("cityNum")
    val indexDf: DataFrame = indexer.fit(df2).transform(df2)
    indexDf.show(false)
    val oneHotEncoder: OneHotEncoder = new OneHotEncoder().setInputCol("cityNum").setOutputCol("cityMatrix").setDropLast(false)
    oneHotEncoder.transform(indexDf).show(false)
  }
}
