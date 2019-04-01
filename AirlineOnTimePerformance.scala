package com.mapr.mlib

import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.Params
import org.apache.spark.storage.StorageLevel
import org.apache.spark.ml.feature.{ OneHotEncoder, StringIndexer, Bucketizer }
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.log4j._
import org.apache.spark.sql._

import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.tuning._
import org.apache.spark.ml.Pipeline

import java.nio.ByteBuffer;

import ConversionsData.toInt
import ConversionsData.get_int_hour

object AirlineOnTimePerformance extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
  lazy val sparkConf = new SparkConf()
    .setAppName("Learn Spark")
    .setMaster("local[*]")

    .set("spark.executor.memory", "70g")
    .set("spark.memory.fraction", "1")
    .set("spark.driver.memory", "50g")
    .set("spark.memory.offHeap.size", "16g");

  val spark = SparkSession
    .builder()
    .config(sparkConf)
    .getOrCreate()

  println(spark.sparkContext.getConf.toDebugString);

  val rdd2007Count = spark.sparkContext.textFile("hdfs://localhost:8020/emirates/data/otp/2007.csv").getNumPartitions

  val rdd2007 = spark.sparkContext.textFile("hdfs://localhost:8020/emirates/data/otp/2007.csv").repartition(7000);
  println(rdd2007Count + " rdd2007");

  val rdd2008 = spark.sparkContext.textFile("hdfs://localhost:8020/emirates/data/otp/2008.csv").repartition(7000);

  import spark.implicits._

  val header = rdd2007.first()

  val flightsds = rdd2007.filter(x => x != header)

  val header1 = rdd2008.first()

  val flights2008ds = rdd2008.filter(x => x != header)

  val flightsdF2007 = flightsds.map(parseFlight).toDS().limit(100000)

  val flightsdF2008 = flights2008ds.map(parseFlight).toDS().limit(100000).na.drop();

  val flightsdF2007Subset = flightsdF2007.filter("DepDelay > 0").na.drop();

  flightsdF2007.describe("UniqueCarrier", "TailNum", "Origin", "Dest", "DayOfWeek", "CRSDepHour",
    "CRSElapsedTime", "CRSArrTime", "CRSDepTime", "Distance").show()

  flightsdF2008.describe("UniqueCarrier", "TailNum", "Origin", "Dest", "DayOfWeek", "CRSDepHour",
    "CRSElapsedTime", "CRSArrTime", "CRSDepTime", "Distance").show()

  val delaybucketizer = new Bucketizer().setInputCol("DepDelay")
    .setOutputCol("delayed").setSplits(Array(0.0, 40.0, Double.PositiveInfinity))

  val flightsdF2007Delay = delaybucketizer.transform(flightsdF2007Subset)
  flightsdF2007Delay.groupBy("delayed").count.show
  flightsdF2007Delay.createOrReplaceTempView("flight")

  println("what is the count of departure delay and not delayed by origin")
  spark.sql("select Origin, delayed, count(delayed) from flight group by Origin, delayed order by Origin").show

  println("what is the count of departure delay by dest")

  spark.sql("select Dest, delayed, count(delayed) from flight where delayed=1 group by Dest, delayed order by Dest").show

  println("what is the count of departure delay by origin, dest")

  spark.sql("select Origin,Dest, delayed, count(delayed) from flight where delayed=1 group by Origin,Dest, delayed order by Origin,Dest").show

  println("what is the count of departure delay by dofW")
  spark.sql("select DayOfWeek, delayed, count(delayed) from flight where delayed=1 group by DayOfWeek, delayed order by DayOfWeek").show

  println("what is the count of departure delay by hour where delay minutes >40")
  spark.sql("select CRSDepTime, delayed, count(delayed) from flight where delayed=1 group by CRSDepTime, delayed order by CRSDepTime").show

  println("what is the count of departure delay carrier where delay minutes >40")
  spark.sql("select UniqueCarrier, delayed, count(delayed) from flight where delayed=1 group by UniqueCarrier, delayed order by UniqueCarrier").show

  val fractions = Map(0.0 -> .29, 1.0 -> 1.0)
  val strain = flightsdF2007Delay.stat.sampleBy("delayed", fractions, 36L)
  strain.groupBy("delayed").count.show

  val splits = strain.randomSplit(Array(0.7, 0.3))
  val (train, test) = (splits(0), splits(1))

  val categoricalColumns = Array("UniqueCarrier", "TailNum", "Origin", "Dest")

  val stringIndexers = categoricalColumns.map { colName =>
    new StringIndexer()
      .setInputCol(colName)
      .setOutputCol(colName + "Indexed")
      .fit(flightsdF2007Subset)
  }
  val encoders = categoricalColumns.map { colName =>
    new OneHotEncoder()
      .setInputCol(colName + "Indexed")
      .setOutputCol(colName + "Enc")
  }

  val labeler = new Bucketizer().setInputCol("DepDelay")
    .setOutputCol("label")
    .setSplits(Array(0.0, 40.0, Double.PositiveInfinity))
  val featureCols = Array("UniqueCarrierEnc", "TailNumEnc", "OriginEnc",
    "DestEnc", "DayOfWeek", "CRSDepHour", "CRSElapsedTime", "CRSArrTime", "CRSDepTime", "Distance")
  //put features into a feature vector column   
  val assembler = new VectorAssembler()
    .setInputCols(featureCols)
    .setOutputCol("features")

  val dTree = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features").setMaxBins(7000)
  val steps = stringIndexers ++ encoders ++ Array(labeler, assembler, dTree)

  val pipeline = new Pipeline().setStages(steps)

  val paramGrid = new ParamGridBuilder().addGrid(dTree.maxDepth, Array(4, 5, 6)).build()

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label").setPredictionCol("prediction")
    .setMetricName("accuracy")

  // Set up 3-fold cross validation with paramGrid
  val crossval = new CrossValidator().setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(3);

  val ntrain = train.drop("delayed").drop("ArrDelay").drop("Year").drop("Month").drop("DayofMonth").drop("DepTime").drop("ArrTime").drop("FlightNum")
    .drop("ActualElapsedTime").drop("AirTime").drop("TaxiIn").drop("TaxiOut").drop("Cancelled").drop("CancellationCode").drop("Diverted").drop("CarrierDelay")
    .drop("WeatherDelay").drop("NASDelay").drop("SecurityDelay").drop("LateAircraftDelay");

  val testDrop = test.drop("ArrDelay").drop("Year").drop("Month").drop("DayofMonth").drop("DepTime").drop("ArrTime").drop("FlightNum")
    .drop("ActualElapsedTime").drop("AirTime").drop("TaxiIn").drop("TaxiOut").drop("Cancelled").drop("CancellationCode").drop("Diverted").drop("CarrierDelay")
    .drop("WeatherDelay").drop("NASDelay").drop("SecurityDelay").drop("LateAircraftDelay");

  testDrop.show
  println(ntrain.count)
  ntrain.show
  val cvModel = crossval.fit(ntrain)

  val predictions = cvModel.transform(test)
  println("after Model");
  predictions.show();

  val accuracy = evaluator.evaluate(predictions)

  val lp = predictions.select("label", "prediction")
  val counttotal = predictions.count()
  val label0count = lp.filter($"label" === 0.0).count()
  val pred0count = lp.filter($"prediction" === 0.0).count()
  val label1count = lp.filter($"label" === 1.0).count()
  val pred1count = lp.filter($"prediction" === 1.0).count()

  val correct = lp.filter($"label" === $"prediction").count()
  val wrong = lp.filter(not($"label" === $"prediction")).count()
  val ratioWrong = wrong.toDouble / counttotal.toDouble
  val ratioCorrect = correct.toDouble / counttotal.toDouble
  val truep = lp.filter($"prediction" === 0.0)
    .filter($"label" === $"prediction").count() / counttotal.toDouble
  val truen = lp.filter($"prediction" === 1.0)
    .filter($"label" === $"prediction").count() / counttotal.toDouble
  val falsep = lp.filter($"prediction" === 0.0)
    .filter(not($"label" === $"prediction")).count() / counttotal.toDouble
  val falsen = lp.filter($"prediction" === 1.0)
    .filter(not($"label" === $"prediction")).count() / counttotal.toDouble

  println("ratio correct", ratioCorrect)

  cvModel.write.overwrite().save("hdfs://localhost:8020//emirates/data/modelled1")

  case class Flight(Year: Int, //1
                    Month: Int, //2
                    DayofMonth: Int, //3
                    DayOfWeek: Int, //4
                    DepTime: Int, //5
                    CRSDepTime: Int, //6
                    CRSDepHour: Int, //6
                    ArrTime: Int, //7
                    CRSArrTime: Int, //8
                    UniqueCarrier: String, // 9 unique carrier code 
                    FlightNum: Int, // 10 flight number
                    TailNum: String, // 11 plane tail number
                    ActualElapsedTime: Int, //12
                    CRSElapsedTime: Int, //13
                    AirTime: Int, //14
                    ArrDelay: Int, //15
                    DepDelay: Int, //16
                    Origin: String, // 17 origin IATA airport code
                    Dest: String, // 18 destination airport code  
                    Distance: Int, // 19 in miles
                    TaxiIn: Int, //20
                    TaxiOut: Int, //21
                    Cancelled: Int, //22 was the flight cancelled
                    CancellationCode: String, // 23 reason for cancellation (A = carrier, B = weather, C = NAS, D = security)
                    Diverted: Int, //  24 1 = yes, 0 = no
                    CarrierDelay: Int, //  25 in minutes
                    WeatherDelay: Int, //  26  in minutes
                    NASDelay: Int, //   27 in minutes
                    SecurityDelay: Int, // 28  in minutes
                    LateAircraftDelay: Int) //29

  def parseFlight(str: String) = {
    val row = str.split(",")
    Flight(
      toInt(row(0)), toInt(row(1)), toInt(row(2)), toInt(row(3)), toInt(row(4)), toInt(row(5)), get_int_hour(toInt(row(5))), toInt(row(6)),
      toInt(row(7)), row(8).trim, toInt(row(9)), row(10).trim, toInt(row(11)), toInt(row(12)),
      toInt(row(13)), toInt(row(14)), toInt(row(15)), row(16).trim,
      row(17).trim, toInt(row(18)), toInt(row(19)), toInt(row(20)),
      toInt(row(21)), row(22).trim, toInt(row(23)), toInt(row(24)),
      toInt(row(25)), toInt(row(26)), toInt(row(27)), toInt(row(28)))

  }

}