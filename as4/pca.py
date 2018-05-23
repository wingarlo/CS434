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
    result = np.zeros(784)
    n = len(x)
    for point in x:
        result+=point
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



    
data = readData("./data/data-1.txt")
center = center(data)
#covariance = covariance(data,center)
print(center)
print(len(center))

#print(covariance)

plt.imshow(np.reshape(center,(28,28)))
plt.show()