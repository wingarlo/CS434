import sys
import numpy as np
import os
import math
from random import randint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

def center(x): #computes center of data
    result = x[0]
    n = len(x)
    for i in range(1,n):
        result+=x[i]
    result = result/n
    return result

def covariance(x,c):
    result = np.zeros((784,784))
    n = len(x)
    for point in x:
        variance = np.dot((point-c),(point-c).T)
        result = np.add(result,variance)
    result = result/n
    return result

def findEigenvectors(covarimatrix):
	values, vectors = numpy.linalg.eig(covarimatrix)
	index = 0
	ranking = 0
	sortValues = []
	for i in values:
		sortValues.append((i,index))
		index += 1
	sortValues = sorted(sortValues, key=lambda eigen: eigen[0], reverse=true)
	topTen = sortValues[:10]
	tenVectors = np.array([vectors[topTen[0][1]],vectors[topTen[1][1]],vectors[topTen[2][1]],vectors[topTen[3][1]],vectors[topTen[4][1]],vectors[topTen[5][1]],vectors[topTen[6][1]],vectors[topTen[7][1]],vectors[topTen[8][1]],vectors[topTen[9][1]]])
    return topTen, tenVectors
    
data = readData("./data/data-1.txt")
center = center(data)
#covariance = covariance(data,center)
print(center)
print(len(center))

#print(covariance)

plt.imshow(np.reshape(center,(28,28)))
plt.show()