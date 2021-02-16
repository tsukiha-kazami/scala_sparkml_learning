package feature.extractor

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.Vector
/**
 * @author Shi Lei
 * @create 2021-02-13
 */
//文字转化成空间向量
object Work2VectorL {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]")
      .appName("ml").getOrCreate()
    spark.sparkContext.setLogLevel("warn")


    // Input data: Each row is a bag of words from a sentence or document.
    val documentDF = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")

    documentDF.show(false)

    // Learn a mapping from words to Vectors.
    val word2Vec: Word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(6)//向量维度数量
      .setMinCount(0)//至少在多少文档中出现

    val model: Word2VecModel = word2Vec.fit(documentDF)

    val result: DataFrame = model.transform(documentDF)

    result.show(false)
    println("-----------------")
    result.collect().foreach { case Row(text: Seq[_], features: Vector) =>
      println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n") }
  }
}
