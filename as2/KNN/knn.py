import numpy as np
import math
import sys
import csv

def loadCSV(fileName):
	nFeat = 30
	xload = []
	yload = []
	temp = []
	with open(fileName) as f:
		string = f.read().replace('\n',',')
		lines = string.split(',')									#lines = list of every data point
	lines = lines[:-1]
	for count in range (0,len(lines)):
		if (count+1) % (nFeat+1) == 1:								#if position is 1(y value) append to y
			yload.append([float(lines[count])])
		else:														#else (if first position append a dummy '1') append to temp
			temp.append(float(lines[count]))
	xload = [temp[i:i+nFeat] for i in range(0,len(temp),nFeat)]		#splitting x into a list of lists separated by each instance
	x1 = np.matrix(xload)											#converting x from list of lists to matrices
	y1 = np.matrix(yload)											#converting y from list of lists to matrices
	print x1
	print y1
	return x1,y1
	
loadCSV("knn_test.csv")