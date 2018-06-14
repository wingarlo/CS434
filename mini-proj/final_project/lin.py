import csv
import numpy as np
from loadingdata import *

def normalize(x):
	return (1 / (1 + np.exp(-x)))


file = "./data/Subject_1.csv"
a,b,c = loadCSV(file, "./data/list_1.csv")
x,y = compileHalfHour(a,b,c)		#load training data

file = "./data/Subject_4.csv"
a,b,c = loadCSV(file, "./data/list_4.csv")
x1,y1 = compileHalfHour(a,b,c)		#load training data

file = "./data/Subject_6.csv"
a,b,c = loadCSV(file, "./data/list_6.csv")
x2,y2 = compileHalfHour(a,b,c)		#load training data

file = "./data/Subject_9.csv"
a,b,c = loadCSV(file, "./data/list_9.csv")
x3,y3 = compileHalfHour(a,b,c)		#load training data

x = np.concatenate((x, x1, x2, x3), axis=0)
y = np.concatenate((y, y1, y2, y3))

x = np.matrix(x)
y = np.matrix(y)
y=y.T
w = np.matrix([])
temp = (x.T * x)
temp = temp.I
temp2 = temp * x.T
w = temp2 * y			#weight = (x^T * x)^-1 * x^T * y
file = "./data/general_test_instances.csv"
x = compileTestdata(file)		#load testing data
x = np.matrix(x)

results = []

for i in range (0,len(x)):
	a = 0.
	b = 0
	a = (float(x[i]*w))
	if a < 0:
		a = 0
	if a > 1:
		a = 1
	if a >= 0.5:
		b = 1
	results.append([a,b])

np.savetxt("general_pred2.csv", results, delimiter=",")