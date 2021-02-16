package classification

import com.sun.org.apache.xml.internal.utils.StringVector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
 * @author Shi Lei
 * @create 2021-02-13
 */
//决策树
object DecisionTreeL {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]")
      .appName("ml").getOrCreate()
    spark.sparkContext.setLogLevel("warn")
    import spark.implicits._

    val dataFrame: DataFrame = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv("dataset/iris.csv").toDF()
    dataFrame.show(false)

    //拆分成 训练模型和测试模型
    val Array(train,test): Array[Dataset[Row]] = dataFrame.randomSplit(Array(0.8, 0.2),10)
//    train.show(false)

    val nameIndex: StringIndexerModel = new StringIndexer().setInputCol("species").setOutputCol("nameIndex").fit(dataFrame)
    //将特赠值合并
    val vectorAssembler: VectorAssembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width")).setOutputCol("features")
//    vectorAssembler.transform(dataFrame).show(false)

    //输出内容转化器
    val predConverter: IndexToString = new IndexToString().setInputCol("prediction").setOutputCol("predlabel").setLabels(nameIndex.labels)

    //创建决策树分类器
    val treeClassifier: DecisionTreeClassifier = new DecisionTreeClassifier()
      .setLabelCol("nameIndex").setFeaturesCol("features")
      .setMaxDepth(8)//设置树的最大深度

    val pipeline: Pipeline = new Pipeline().setStages(Array(nameIndex, vectorAssembler, treeClassifier, predConverter))
    val model: PipelineModel = pipeline.fit(train)
    val preds: DataFrame = model.transform(test)
    //查询所有
//    preds.show(false)
    //选出10个数量进行查询
//    preds.select("species","predlabel").show(10)


    //模型的评估

    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator().setLabelCol("nameIndex").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy: Double = evaluator.evaluate(preds)

    println("accuracy: " +accuracy*100 +"%")

    preds.printSchema()





  }
}
