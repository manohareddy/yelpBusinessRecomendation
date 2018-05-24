from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.mllib.recommendation import ALS
from time import time
import math
import json

spark = SparkSession \
        .builder \
        .appName("yelp") \
        .getOrCreate()
   
sc = spark.sparkContext

start1=time()

ratings = sc.textFile('/data2/zombiefiles/dataset/review.json')\
          .map(json.loads)\
          .map(lambda x:(x['user_id'], x['business_id'], x['stars']))\
          .cache()

business = ratings.map(lambda x:x[1])\
                  .distinct()\
                  .zipWithIndex()\
                  .map(lambda x:Row(b_id = x[0], b_ix = x[1]))\
                  .toDF()\
                  .createOrReplaceTempView('bdata')

users = ratings.map(lambda x:x[0])\
               .distinct()\
               .zipWithIndex()\
               .map(lambda x:Row(u_id = x[0], u_ix = x[1]))\
               .toDF()\
               .createOrReplaceTempView('udata')

rating = spark.read.json('/data2/zombiefiles/dataset/review.json')\
                   .createOrReplaceTempView('data')

ei = time()

ratData= spark.sql('select data.stars, udata.u_ix as u, bdata.b_ix as b from data, udata, bdata where data.user_id = udata.u_id and data.business_id = bdata.b_id')\
                  .rdd\
                  .map(lambda x:(x[1],x[2],x[0]))

training_RDD, validation_RDD, test_RDD = ratData.randomSplit([6, 2, 2], seed=0L)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

seed = 5L
iterations = 25
regularization_parameter = 0.2
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02
 
min_error = float('inf')
best_rank = -1
best_iteration = -1

starti=time()

for rank in ranks:
    timeST=time()
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    print 'took',(time()-timeST)
    if error < min_error:
        min_error = error
        best_rank = rank

print 'The best model was trained with rank %s' % best_rank
end = time()
print 'Time taken to build the model -',(end-start1)


# testing

training_RDD, test_RDD = ratData.randomSplit([7, 3], seed=0L)

complete_model = ALS.train(training_RDD, best_rank, seed=seed,iterations=iterations, lambda_=regularization_parameter)

predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print 'Testing data the RMSE is %s' %(error)

#gathering business ids

spark.read.json('/data2/zombiefiles/dataset/business.json').createOrReplaceTempView('business')

complete_businesses_data = spark.sql('select b_ix, b_id, name from bdata, business where bdata.b_id = business.business_id').rdd

complete_business_titles = complete_businesses_data.map(lambda x: (int(x[0]),x[2]))

#new user creation(dummy)

new_user_ratings = [(2133121212, 2345, 4), (2133121212, 3455, 4),(2133121212, 35345, 5),(2133121212, 5645, 4),(2133121212, 8345, 4),(2133121212, 985, 3),(2133121212, 23345, 3),(2133121212, 145, 5),(2133121212, 2545, 4),(2133121212, 234335, 4),(2133121212, 45, 2),(2133121212, 235, 3),(2133121212, 25, 5)]

new_user_ratings_RDD = sc.parallelize(new_user_ratings)

complete_data_with_new_ratings_RDD = ratData.union(new_user_ratings_RDD)

t0 = time()
new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=seed, 
                              iterations=iterations, lambda_=regularization_parameter)
tt = time() - t0
#print
print "New model trained in %s seconds" % round(tt,3)

new_user_ratings_ids = new_user_ratings_RDD.map(lambda x: x[1])

new_user_ids = new_user_ratings_ids.collect()

new_user_unrated_businesses_RDD = complete_business_titles.filter(lambda x: x[0] not in new_user_ids).map(lambda x: (2133121212, x[0]))

new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_businesses_RDD)

new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))

new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_RDD.join(complete_business_titles)

print new_user_recommendations_rating_RDD.sortBy(lambda x:x[1], ascending=False).take(20)


