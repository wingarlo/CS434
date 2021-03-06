import sys
import numpy as np
from numpy import linalg as LA
import os
import math
from random import randint

import matplotlib as mpl
mpl.use('GTKAgg')
from matplotlib import pyplot as plt

dim = 784

def dist(p1, p2):#takes in 2 arrays of any length and returns distance
    distance = 0
    for i in range(0,len(p1)):
        distance += float((float(p1[i])-float(p2[i]))**2)
    return math.sqrt(distance)

def readData(filename):
    x = []
    with open(filename) as f:
        for lines in f:
            lines = lines[:-1]#getting rid of \n
            x.append(lines.split(','))
    x = np.array(x)
    return x.astype(np.int)#convert from string to int

def findEigen(covarimatrix):
	values, vectors = LA.eig(covarimatrix)
	index = 0
	ranking = 0
	sortValues = []
	for i in values:
		sortValues.append((i,index))
		index += 1
	sortValues = sorted(sortValues, key=lambda eigen: eigen[0], reverse=True)
	topTen = sortValues[:10]
	topTenA = np.array([values[topTen[0][1]],values[topTen[1][1]],values[topTen[2][1]],values[topTen[3][1]],values[topTen[4][1]],values[topTen[5][1]],values[topTen[6][1]],values[topTen[7][1]],values[topTen[8][1]],values[topTen[9][1]]])
	tenVectors = np.array([vectors[topTen[0][1]],vectors[topTen[1][1]],vectors[topTen[2][1]],vectors[topTen[3][1]],vectors[topTen[4][1]],vectors[topTen[5][1]],vectors[topTen[6][1]],vectors[topTen[7][1]],vectors[topTen[8][1]],vectors[topTen[9][1]]])
	return topTenA,tenVectors


    
data = readData("./data/data-1.txt")
dataT = data.T
center = np.mean(data, axis=0)
covar = np.cov(dataT)
print "center"
print(center)
print "covariance array"
print(covar)
print "covariance array shape"
print(covar.shape)

#plotting mean image
plt.imshow(np.reshape(center,(28,28)))
#plt.show()

eigVals,eigVec = findEigen(covar)
eigVals = eigVals.real
eigVec = eigVec.real

id = 0
for v in eigVec:
	v = np.multiply(v, eigVals[id])
	id += 1

print "Top Ten Eigenvalues"
print eigVals
print "Top Ten Eigenvectors"
print eigVec

projections = eigVec.dot(data.T)
projections = projections.T
print projections.shape

dimMax = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
dimPeakID = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
id = 0
for p in projections:
	eid = 0
	for ex in p:
		if ex > dimMax[eid]:
			dimMax[eid] = ex
			dimPeakID[eid] = id
		eid += 1
	id += 1

print "dimMax"
print dimMax
print "dimPeakID"
print dimPeakID


filename = 'eigen/eigenValues.csv'
np.savetxt(filename, eigVals, delimiter=",")

fig=plt.figure(figsize=(8, 8))
columns = 5
rows = 2

#plotting eigen vectors
vectorcount = 0
for q in range(len(eigVec)):
	filename = 'eigen/eigenVector'+str(vectorcount)+'.csv'
	np.savetxt(filename, eigVec[vectorcount], delimiter=",")
	fig.add_subplot(rows, columns, q)
	plt.imshow(np.reshape(eigVec[vectorcount],(28,28)))
	vectorcount += 1
#plt.show()

#plotting top projections
count = 0
for q in range(len(dimPeakID)):
	filename = 'bestMatch/data'+str(count)+'.csv'
	dataID = dimPeakID[count]
	np.savetxt(filename, data[dataID], delimiter=",")
	fig.add_subplot(rows, columns, q)
	plt.imshow(np.reshape(data[dataID],(28,28)))
	count += 1
plt.show()