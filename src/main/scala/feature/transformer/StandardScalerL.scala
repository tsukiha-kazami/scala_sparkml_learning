package feature.transformer

import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
 * @author Shi Lei
 * @create 2021-02-13
 */
//标准化处理
object StandardScalerL {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]")
      .appName("ml").getOrCreate()
    spark.sparkContext.setLogLevel("warn")
    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -1.0)),
      (1, Vectors.dense(2.0, 1.1, 2.0)),
      (2, Vectors.dense(3.0, 10.1, 0.0))
    )).toDF("id", "features")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true) //是否要除以标准差
      .setWithMean(true) //是否要将平均值放到中间



    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(dataFrame)


    // Normalize each feature to have unit standard deviation.
    val scaledData = scalerModel.transform(dataFrame)
    scaledData.show(false)
  }
}
