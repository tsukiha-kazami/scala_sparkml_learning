package recommend

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.jblas.DoubleMatrix

/**
 * @author Shi Lei
 * @create 2021-02-16
 */

object ContentRecommender {

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("ContentRecommender")

    // 创建一个SparkSession
    val spark: SparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._


    val reader = new DataReader(spark)
    val movieDf = reader.readMovieDf()

    val movieTagsDF: DataFrame = movieDf.as[Movie].map(
      x => (x.mid, x.name, x.genres.map(c => if (c == '|') ' ' else c)) // Action|Adventure|Sci-Fi
    )
      .toDF("mid", "name", "genres")
      .cache()//緩存


    // 用tf-idf 从内容信息中提取电影的特征向量
    val tokenizer = new Tokenizer().setInputCol("genres").setOutputCol("words")
    // 对文本进行分词处理, 按照空格
    val wordsData = tokenizer.transform(movieTagsDF)
    wordsData.show(truncate = false)

    // 计算词频 TF
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(50).setBinary(false)
    val featuredData = hashingTF.transform(wordsData)
    featuredData.show(truncate = false)

    // 计算tf-idf
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    // 训练idfmodel
    val idfmodel = idf.fit(featuredData)
    val resclaledData = idfmodel.transform(featuredData)
//    resclaledData.show(truncate = false)

    // 两两之间计算电影特征的相似度 , 做计算之前, 需要讲向量数据展开
    val movieFeatures = resclaledData.map(
      row => (row.getAs[Int]("mid"), row.getAs[SparseVector]("features").toArray)
    )
      .rdd
      .map(
        x => (x._1, new DoubleMatrix(x._2))
      )

    //输出
    //    movieFeatures.collect().foreach(println)
    // 自己和自己做笛卡尔积 肯定会有一组数据是自己和自己的配对, 自己和自己的配对计算得到的相似度肯定为1
    val moviewRecs = movieFeatures.cartesian(movieFeatures)
      .filter{
        // 过滤自己和自己的配对
        case (a, b) => a._1 != b._1
      }
      .map{
        // 计算相似度
        case (a,b) => {
          val simScore = consinSim(a._2, b._2)
          (a._1, (b._1, simScore))
        }
      }
      .filter(_._2._2 > 0.7) // 只保存相似度大于0.7
      .groupByKey() // 根据mid分组
      .map{
        case(mid, items) => MovieRecs(mid, items.toList.sortWith(_._2 > _._2).take(20).map(x => Recommendation(x._1, x._2)))
      }
      .toDF()
    moviewRecs.show(false)
    spark.stop()
  }

  // 计算电影之间的余弦相似度
  def consinSim(mv1: DoubleMatrix, mv2: DoubleMatrix): Double = {
    mv1.dot(mv2) / (mv1.norm2() * mv2.norm2())
  }
}




