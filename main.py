from pyspark import SparkContext
from pyspark.sql import SparkSession

SparkContext.setSystemProperty('spark.storage.memoryFraction', '0') # if no cache() and/or persist().
SparkContext.setSystemProperty('spark.executor.memory', '2g')
# SparkContext.setSystemProperty('spark.driver.memory', '6g')

sc = SparkContext(appName='spectral')
spark = SparkSession(sc)

from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import sqrt

import sys


K = 5

## Read data.
txt = sc.textFile('./data/com-amazon.ungraph.txt')
txt = txt.sample(False, 0.001, 1) # XXX: random sample for local testing
txt = txt.zipWithIndex().filter(lambda x: int(x[1]) >= 4).map(lambda x: x[0].split('\t'))

## Get graph Laplacian
N = txt.flatMap(lambda x: [int(xx) for xx in x]).max()

upper_entries = txt.map(lambda x: MatrixEntry(int(x[0])-1, int(x[1])-1, 1.0))
lower_entries = txt.map(lambda x: MatrixEntry(int(x[1])-1, int(x[0])-1, 1.0))
degrees = upper_entries.map(lambda entry: (entry.i, entry.value)).reduceByKey(lambda a, b: a + b)
W = CoordinateMatrix(upper_entries.union(lower_entries), numCols=N, numRows=N)

# XXX:
laplacian = sys.argv[1]

if laplacian == 'unnormalized':
    entries = degrees.map(lambda x: MatrixEntry(x[0], x[0], x[1]))
    D = CoordinateMatrix(entries, numCols=N, numRows=N)
    L = D.toBlockMatrix().subtract(W.toBlockMatrix()).toCoordinateMatrix()
elif laplacian == 'normalized':
    entries = degrees.map(lambda x: MatrixEntry(x[0], x[0], 1/x[1]))
    D_inv = CoordinateMatrix(entries, numCols=N, numRows=N).toBlockMatrix()
    I = CoordinateMatrix(sc.range(N).map(lambda i: MatrixEntry(i, i, 1.0)), numCols=N, numRows=N).toBlockMatrix()
    L = I.subtract(D_inv.multiply(W.toBlockMatrix())).toCoordinateMatrix()
elif laplacian == 'symmetric':
    entries = degrees.map(lambda x: MatrixEntry(x[0], x[0], 1/sqrt(x[1])))
    D_invsq = CoordinateMatrix(entries, numCols=N, numRows=N).toBlockMatrix()
    I = sc.range(N).map(lambda i: MatrixEntry(i, i, 1.0), N, N)
    tmp = D_invsq.multiply(W.toBlockMatrix()).multiply(D_invsq)
    L = I.toBlockMatrix().subtract(tmp)
else:
    raise ValueError('Unknown type of Laplacian.')


## SVD, and transform from dense matrix to dataframe.
svd = L.toRowMatrix().computeSVD(k=K, computeU=False)
V = svd.V.toArray().tolist()
VV = spark.createDataFrame(V)
kmeans = KMeans().setK(K).setSeed(1)
vecAssembler = VectorAssembler(inputCols=VV.schema.names, outputCol='features')
VV = vecAssembler.transform(VV)

## Kmeans
model = kmeans.fit(VV.select('features'))
clusters = model.transform(VV)

## Save results
clusters.toPandas().to_csv('./out/assignment.csv')


## Stop context
sc.stop()
