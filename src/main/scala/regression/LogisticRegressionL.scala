package regression

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, LogisticRegressionTrainingSummary}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel, LinearRegressionTrainingSummary}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * @author Shi Lei
 * @create 2021-02-16
 */
object  LogisticRegressionL {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]")
      .appName("ml").getOrCreate()
    spark.sparkContext.setLogLevel("warn")

    val dataFrame: DataFrame = spark.read.format("libsvm").load("dataset/libsvm.txt")
//    dataFrame.show(false)


    val regression = new LogisticRegression()
      .setMaxIter(10)//设置最大迭代次数，默认梯度下降
      .setRegParam(0.3)//L2正则化系数, 高次项之前乘以该系数
      .setElasticNetParam(0.8)//设置梯度下降的步长

    val model: LogisticRegressionModel = regression.fit(dataFrame)
    println(s"个项的权重系数，${model.coefficients} 偏置：${model.intercept}")

    val summary: LogisticRegressionTrainingSummary = model.binarySummary

    println(s"迭代次數，${summary.totalIterations}")
    println(s"迭代计算历史误差，${summary.objectiveHistory.mkString(",")}")
  }
}
