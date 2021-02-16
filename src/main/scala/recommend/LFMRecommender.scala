package recommend

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.jblas.DoubleMatrix

/**
 * @author Shi Lei
 * @create 2021-02-16
 */
object LFMRecommender {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("LFMRecommender")
    // 创建一个SparkSession
    val spark: SparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    spark.sparkContext.setLogLevel("warn")
    import spark.implicits._

    val reader = new DataReader(spark)
    val ratingDf: DataFrame = reader.readRatingDF()


    val ratingRDD: RDD[(Int, Int, Double)] = ratingDf.limit(100).as[MoiveRating].rdd.map(r => (r.uid, r.mid, r.score))
//      .cache()
//    ratingRDD.foreach(println)

    // 准备训练数据 ALS uid, product_id, rating Rating
    val trainData = ratingRDD.map(x => Rating(x._1, x._2, x._3))
    val model = ALS.train(trainData,20, 10, 0.01)//rank是维度数量;iterations为最大迭代数;lambda为正则化惩罚系数

    // 预测用户对电影的评分, 准备测试数据
    // 获取去重之后的用户id 和去重之后的mid
    val useRDD = ratingRDD.map(_._1).distinct()
    val movieRDD = ratingRDD.map(_._2).distinct()
    // 笛卡尔积
    val userMovies = useRDD.cartesian(movieRDD)
    userMovies.collect().foreach(println)


    val predict = model.predict(userMovies)
    predict.collect.foreach(println)


    val user_recs = predict
      .filter(_.rating > 0) // 过滤掉小于0的评分数据
      .map(rating=> (rating.user, (rating.product, rating.rating)))
      .groupByKey()
      .map{
        case(uid, recs) => UserRecs(uid, recs.toList.sortWith(_._2 > _._2).take(20).map(x => Recommendation(x._1, x._2)))
      }
      .toDF()
    user_recs.show(false)


    // 根据隐语义模型挖掘的物品的隐藏特征, 计算物品的相似度, 提前给实时推荐做准
    model.userFeatures//用户相似度
    model.productFeatures//商品相似度
    val movieFeatures = model.productFeatures.map{
      case(mid, features) => (mid, new DoubleMatrix(features))
    }

    val movieRecs = movieFeatures.cartesian(movieFeatures)
      .filter{
        case(a, b) => a._1 != b._1
      }.map{
      case(a, b) => {
        val simScore = consinSim(a._2, b._2)
        (a._1, (b._1, simScore))
      }
    }
      .filter(_._2._2 > 0.7)
      .groupByKey()
      .map{
        case(mid, items) => MovieRecs(mid, items.toList.sortWith(_._2 > _._2).map(x => Recommendation(x._1, x._2)))
      }
      .toDF()

    movieRecs.show(false)

  }

  def consinSim(mv1: DoubleMatrix, mv2: DoubleMatrix): Double = {
    mv1.dot(mv2) / (mv1.norm2() * mv2.norm2())
  }
}
