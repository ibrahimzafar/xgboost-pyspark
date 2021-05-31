
# # #Check which version of Python is running
import sys
import os

# import findspark
# os.environ['PYSPARK_PYTHON'] = sys.executable
# #os.environ['PYSPARK_PYTHON'] = '/home/izafar/.conda/envs/temp_env/bin/python'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# os.environ['SPARK_HOME'] = '/root/park-3.1.1-bin-hadoop2.7'
# # # os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1' #only if Spark 2.3, 2.4 and PyArrow >=0.15
# # # since we have Spark3.0, we don't need to do anything                                            

# findspark.init() 

# # from pyspark import SparkContext
# from pyspark.sql import SparkSession, SQLContext
# # from pyspark.sql.functions import mean as _mean, stddev as _stddev, col, upper, sin, cos, log
# # from pyspark.conf import SparkConf


# os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# export PYARROW_WITH_CUDA=1
# os.environ['PYARROW_WITH_CUDA'] = '1'

# conf = SparkConf().setAppName("GPU Experiment App")
# conf = (conf.setMaster('local[*]')
#         .set('spark.executor.memory', '20G')
#         .set('spark.driver.memory', '50G')
#         .set('spark.executor.cores', '4'))
# #        .set('spark.driver.maxResultSize', '10G'))
# sc = SparkContext(conf=conf)

# spark = SparkSession(sc)

# print("Spark Version: ", sc.version)
# print(conf.getAll())

# print(os.getcwd())
# print(python_version())
# print(sys.executable)
# ! nproc #number of CPU Cores
# #! lscpu
# #! lshw -C display #Info about GPU
# #Enrironment must be /home/izafar/.conda/envs/spark3cuda_env/bin/python

# 
# http://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html

# Arrow is available as an optimization when converting a Spark DataFrame to a Pandas DataFrame using the call toPandas() and when creating a Spark DataFrame from a Pandas DataFrame with createDataFrame(pandas_df)

# RDD - Resilient Distributed Dataset <br>
# Pandas - dataframe



from sparkxgb import XGBoostClassifier, XGBoostRegressor
from pprint import PrettyPrinter
 
from pyspark.sql.types import StringType
 
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
pp = PrettyPrinter()



from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import mean as _mean, stddev as _stddev, col, upper, sin, cos, log
from pyspark.conf import SparkConf


# In[10]:


# Code taken from https://github.com/sllynn/spark-xgboost/blob/master/examples/spark-xgboost_adultdataset.ipynb
from sparkxgb import XGBoostClassifier, XGBoostRegressor
# from xgboost.spark import XGBoostClassifier
# from xgboost import XGBClassifier

from pprint import PrettyPrinter
 
from pyspark.sql.types import StringType
 
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# In[11]:


from pprint import PrettyPrinter


# In[12]:


# get_ipython().system('pip list | grep ppr')


# In[13]:


# import xgboost as xgb
# xgb.XGBClassifier


# In[14]:


# import os
# [f for f in os.listdir() if 'xgb' in f and '1.4' in f] 
# os.environ['SPARK_HOME']=''

# In[15]:


# spark= SparkSession.builder.config("spark.jars", "xgboost4j-0.90.jar,xgboost4j-spark-0.90.jar").getOrCreate()
spark= SparkSession.builder.config("spark.jars", "xgboost4j-spark-gpu_2.12-1.4.1.jar, xgboost4j_2.12-1.4.1.jar, xgboost4j-spark_2.12-1.4.1.jar").getOrCreate()
# conf = SparkConf().setAppName("Spark_XGB_POC_App")
# conf = (conf.setMaster('local[*]'))
# sc = SparkContext(conf=conf)
# spark = SparkSession(sc)

# 'xgboost4j-spark-gpu_2.12-1.4.1.jar',
#  'xgboost4j-spark_2.12-1.4.1.jar',
#  'xgboost4j_2.12-1.4.1.jar'


# In[16]:


print("Spark Version: ", spark.version)


# In[17]:



col_names = [
  "age", "workclass", "fnlwgt",
  "education", "education-num",
  "marital-status", "occupation",
  "relationship", "race", "sex",
  "capital-gain", "capital-loss",
  "hours-per-week", "native-country",
  "label"
]
 
train_sdf, test_sdf = (
  spark.read.csv(
    path="file:///" + os.getcwd() + "/adult.data",
    inferSchema=True  
  )
  .toDF(*col_names)
  .repartition(200)
  .randomSplit([0.8, 0.2])
)
 
string_columns = [fld.name for fld in train_sdf.schema.fields if isinstance(fld.dataType, StringType)]
string_col_replacements = [fld + "_ix" for fld in string_columns]
string_column_map=list(zip(string_columns, string_col_replacements))
target = string_col_replacements[-1]
predictors = [fld.name for fld in train_sdf.schema.fields if not isinstance(fld.dataType, StringType)] + string_col_replacements[:-1]
pp.pprint(
  dict(
    string_column_map=string_column_map,
    target_variable=target,
    predictor_variables=predictors
  )
)
 
si = [StringIndexer(inputCol=fld[0], outputCol=fld[1]) for fld in string_column_map]
va = VectorAssembler(inputCols=predictors, outputCol="features")
pipeline = Pipeline(stages=[*si, va])
fitted_pipeline = pipeline.fit(train_sdf.union(test_sdf))
 
train_sdf_prepared = fitted_pipeline.transform(train_sdf)
train_sdf_prepared.cache()
train_sdf_prepared.count()
 
test_sdf_prepared = fitted_pipeline.transform(test_sdf)
test_sdf_prepared.cache()
test_sdf_prepared.count()
 
xgbParams = dict(
  eta=0.1,
  maxDepth=2,
  missing=0.0,
  objective="binary:logistic",
  numRound=5,
  numWorkers=2
)
 
xgb = (
  XGBoostClassifier(**xgbParams)
  .setFeaturesCol("features")
  .setLabelCol("label_ix")
)
 
model = xgb.fit(test_sdf_prepared)


# In[18]:


model = xgb.fit(test_sdf_prepared)
print(model)



bce = BinaryClassificationEvaluator(
  rawPredictionCol="rawPrediction",
  labelCol="label_ix"
)

roc_test=bce.evaluate(model.transform(test_sdf_prepared))
print(roc_test)



# import mlflow



 
# param_grid = (
#   ParamGridBuilder()
#   .addGrid(xgb.eta, [1e-1, 1e-2, 1e-3])
#   .addGrid(xgb.maxDepth, [2, 4, 8])
#   .build()
# )

# cv = CrossValidator(
#   estimator=xgb,
#   estimatorParamMaps=param_grid,
#   evaluator=bce,#mce,
#   numFolds=5
# )


# best_params = dict(
# eta_best=model.bestModel.getEta(),
# maxDepth_best=model.bestModel.getMaxDepth()
# )
# import mlflow
# import mlflow.spark

# spark_model_name = "best_model_spark"

# print("HELLOOOOOO")
# with mlflow.start_run():
#   model = cv.fit(train_sdf_prepared)
#   best_params = dict(
#     eta_best=model.bestModel.getEta(),
#     maxDepth_best=model.bestModel.getMaxDepth()
#   )
#   mlflow.log_params(best_params)

#   mlflow.spark.log_model(fitted_pipeline, "featuriser")
#   mlflow.spark.log_model(model, spark_model_name)

#   metrics = dict(
#     roc_test=bce.evaluate(model.transform(test_sdf_prepared))
#   )
#   mlflow.log_metrics(metrics)




