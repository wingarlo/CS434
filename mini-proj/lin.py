import csv
import numpy as np
from loadingdata import *




file = "./data/Subject_2_part1.csv"
a,b,c = loadCSV(file, "./data/list2_part1.csv")
x,y = compileHalfHour(a,b,c)		#load training data

x = np.matrix(x)
y = np.matrix(y)
y=y.T
w = np.matrix([])
temp = (x.T * x)
temp = temp.I
temp2 = temp * x.T
w = temp2 * y			#weight = (x^T * x)^-1 * x^T * y
file = "./data/subject2_instances.csv"
x = compileTestdata(file)		#load testing data
print x
x = np.matrix(x)

results = []

for i in range (0,len(x)):
	a = 0.
	b = 0
	a = float(x[i]*w)
	if a >= 0.5:
		b = 1
	results.append([a,b])

np.savetxt("res2.csv", results, delimiter=",")