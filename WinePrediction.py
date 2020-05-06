# Importing Libraries
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import RandomForestClassifier

# Creating A Spark Session
spark_session = SparkSession.builder.master("local").appName("wineQualityPrediction").config("spark.some.config.option","some-value").getOrCreate()

# Reading Dataset
raw_data = spark_session.read.csv('TrainingDataset.csv', header='true', inferSchema='true', sep=';')

# Validation dataset
val_data = spark_session.read.csv('ValidationDataset.csv',header='true', inferSchema='true', sep=';')

# Creating Feature column
feature_columns = [c for c in raw_data.columns if c != 'quality']
f_vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="merged_features")
input_transform = f_vector_assembler.transform(raw_data)
input_transform.cache()

# Converting for ValidationDataset
feature = [c for c in val_data.columns if c != 'quality']
val_ds_vector_assembler = VectorAssembler(inputCols=feature, outputCol="merged_features")
val_ds_tranform = val_ds_vector_assembler.transform(val_data)

# Creating Model
random_forest_classifier = RandomForestClassifier(labelCol="quality", featuresCol="merged_features", numTrees=13)
model = random_forest_classifier.fit(input_transform)

# Validating the model
prediction_performance = model.transform(val_ds_tranform)

# F1 Score
fone_evaluator = MulticlassClassificationEvaluator(
													labelCol="quality", 
													predictionCol="prediction", metricName="f1"
												)
f_one = fone_evaluator.evaluate(prediction_performance)
print("f1 error = %g" % (1.0 - f_one))
f_one_data_transformed = model.transform(val_ds_tranform)
print(fone_evaluator.getMetricName(), 'accuracy :', fone_evaluator.evaluate(f_one_data_transformed))


