from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
import json
from time import time

spark = SparkSession\
        .builder\
        .appName("movievals")\
        .getOrCreate()

sc = spark.sparkContext
st = time()
for part in range(1,11):
    stime = time()
    data = spark.read.json('/data2/zombiefiles/dataset/review.json')\
            .sample(False, float(part)/10, part**2)\
            .createOrReplaceTempView('data')

    spark.sql('select stars, text from data').createOrReplaceTempView('df')

    stopwords = sc.textFile('/data2/zombiefiles/stopwords.txt')\
              .collect()

    text_RDD = spark.sql('select * from df')\
                .rdd\
                .map(lambda x: (x[0],x[1].strip().lower().split()))\
                .map(lambda x: (x[0], [j for j in x[1] if j not in stopwords]))

    i=[1,2,3,4,5]
    top50 = text_RDD.filter(lambda x:x[0] in i)\
            .flatMap(lambda x:[(i,1) for i in x[1]])\
            .reduceByKey(lambda x,y:x+y)\
            .sortBy(lambda x:x[1], ascending = False)\
            .take(250)
    print 'top 50 words with rating ',i, ' = ', top50

    print part*10,'% took ', time() - stime,' secs to run the word count'

print 'total word count took ', time()-st,' secs'

