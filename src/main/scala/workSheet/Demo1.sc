
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature._

val spark: SparkSession = SparkSession.builder().master("local[*]")
  .appName("ml").getOrCreate()

import spark.implicits._

val df: DataFrame = spark.createDataFrame(Seq(
  (0, Array("a", "b", "c","d")),
  (1, Array("a", "b", "c", "c", "d", "a"))
)).toDF("id", "words")

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

//估计器 => fit => 转化器
//CountVectorizer将根据语料库中的词频排序从高到低进行选择，
val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("words").setOutputCol("features")
  //词汇表的最大含量由vocabsize超参数来指定，
  // 超参数minDF，则指定词汇表中的词语至少要在多少个不同文档中出现
  .setVocabSize(5).setMinDF(1).fit(df)

//自定义模型模型
val cvm = new CountVectorizerModel(Array("a", "b", "c")).setInputCol("words").setOutputCol("features")

//transform
cvModel.transform(df).show(false)

cvm.transform(df).show(false)


