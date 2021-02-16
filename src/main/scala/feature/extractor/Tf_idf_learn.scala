package feature.extractor
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

/**
 * @author Shi Lei
 * @create 2021-02-13
 */
//文字在文章中的重要新占比
object Tf_idf_learn {
  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession.builder().master("local[*]")
      .appName("ml").getOrCreate()
    spark.sparkContext.setLogLevel("warn")


    //转化器对象，更具空格分词，英文默认的就是空格分词
    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "Hi I heard about Spark"),
      (0.0, "I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat")
    )).toDF("label", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures")
      .setNumFeatures(20)//设置特征数量

    val featurizedData = hashingTF.transform(wordsData)

    // alternatively, CountVectorizer can also be used to get term frequency vectors

    //idf 是一个估计器对象
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show()
  }
}
