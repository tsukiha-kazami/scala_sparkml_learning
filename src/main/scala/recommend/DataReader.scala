package recommend

import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * @author Shi Lei
 * @create 2021-02-16
 */
class DataReader(spark: SparkSession) {

  // 定义常量
  private val MOVIE_DATA_PATH = "dataset\\recommend\\movies.csv"
  private val RATING_DATA_PATH = "dataset\\recommend\\ratings.csv"
  private val TAG_DATA_PATH = "dataset\\recommend\\tags.csv"


  import spark.implicits._

  def readMovieDf():DataFrame={
    val movieRDD = spark.sparkContext.textFile(path = MOVIE_DATA_PATH)
    val movieDf= movieRDD.map(
      item => {
        val attr = item.split("\\^")
        Movie(attr(0).toInt, attr(1).trim, attr(2).trim, attr(3).trim, attr(4).trim, attr(5).trim, attr(6).trim, attr(7).trim, attr(8).trim, attr(9).trim)
      }
    ).toDF()
    movieDf
  }

  def readRatingDF():DataFrame={
    val ratingRDD = spark.sparkContext.textFile(RATING_DATA_PATH)
    val ratingDF = ratingRDD.map(item => {
      val attr = item.split(",")
      MoiveRating(attr(0).toInt,attr(1).toInt,attr(2).toDouble,attr(3).toInt)
    }).toDF()
    ratingDF
  }

  def readTagDF():DataFrame={
    val tagRDD = spark.sparkContext.textFile(TAG_DATA_PATH)
    //将tagRDD装换为DataFrame
    val tagDF = tagRDD.map(item => {
      val attr = item.split(",")
      Tag(attr(0).toInt,attr(1).toInt,attr(2).trim,attr(3).toInt)
    }).toDF()
    tagDF
  }

}
/**
 * Movie 数据集
 *
 * 260                                         电影ID，mid
 * Star Wars: Episode IV - A New Hope (1977)   电影名称，name
 * Princess Leia is captured and held hostage  详情描述，descri
 * 121 minutes                                 时长，timelong
 * September 21, 2004                          发行时间，issue
 * 1977                                        拍摄时间，shoot
 * English                                     语言，language
 * Action|Adventure|Sci-Fi                     类型，genres
 * Mark Hamill|Harrison Ford|Carrie Fisher     演员表，actors
 * George Lucas                                导演，directors
 *
 */
case class Movie(mid: Int, name: String, descri: String, timelong: String, issue: String,
                 shoot: String, language: String, genres: String, actors: String, directors: String)

/**
 * Rating数据集
 *
 * 1,31,2.5,1260759144
 */
case class MoiveRating(uid: Int, mid: Int, score: Double, timestamp: Int )

/**
 * Tag数据集
 *
 * 15,1955,dentist,1193435061
 */
case class Tag(uid: Int, mid: Int, tag: String, timestamp: Int)
// 定义一个基准推荐对象
case class Recommendation(mid: Int, score: Double)

// 一个电影相似的电影有哪些
case class MovieRecs(mid: Int, recs: Seq[Recommendation])


//
case class UserRecs(uid: Int, recs: Seq[Recommendation])