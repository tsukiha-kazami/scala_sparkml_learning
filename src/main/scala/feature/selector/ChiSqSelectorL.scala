package feature.selector

import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
 * @author Shi Lei
 * @create 2021-02-13
 */
//卡方選擇器
object ChiSqSelectorL {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]")
      .appName("ml").getOrCreate()
      spark.sparkContext.setLogLevel("warn")
      import spark.implicits._

    val df = spark.createDataset(Seq(
      (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
      (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
      (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
    )
    ).toDF("id", "features", "clicked")

    val selector = new ChiSqSelector()
      .setNumTopFeatures(1)
      .setFeaturesCol("features")
      .setLabelCol("clicked")
      .setOutputCol("selectedFeatures")

    val result = selector.fit(df).transform(df)

    println(s"ChiSqSelector output with top ${selector.getNumTopFeatures} features selected")
    result.show()
  }
}
