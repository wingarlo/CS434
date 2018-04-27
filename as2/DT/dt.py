import numpy as np
import math
import sys
import csv
def sortByFeature(x,y,j):
	for i in range(1, len(y)):
		k = i-1 
		keyx = x[i,j]
		keyy = y[i,0]
		while (x[k,j] > keyx) and (k >= 0):
			x[k+1] = x[k]
			y[k+1,0] = y[k,0]
			k -= 1
		x[k+1,j] = keyx
		y[k+1,0] = keyy
	return x,y
	
def entrop(occur, tot):
	occur = float(occur)
	tot = float(tot)
	if occur == tot:
		return 0 - ((occur/tot) * math.log(occur/tot))
	if occur == 0:
		return (((tot - occur)/tot) * math.log((tot-occur)/tot))
	return (((tot - occur)/tot) * math.log((tot-occur)/tot)) - ((occur/tot) * math.log(occur/tot))
	
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
		else:														#else append to temp
			temp.append(float(lines[count]))
	xload = [temp[i:i+nFeat] for i in range(0,len(temp),nFeat)]		#splitting x into a list of lists separated by each instance
	x1 = np.matrix(xload)											#converting x from list of lists to matrices
	y1 = np.matrix(yload)										#converting y from list of lists to matrices
	x1 = x1 / x1.max(axis=0)										#normalize
	return x1,y1
	
	
def DT():
	maxGain = [0,0,0]
	x,y = loadCSV("knn_train.csv")
	parent = np.count_nonzero(y == 1)				#number of 1's in y
	pEntrop = entrop(parent, len(y))
	for j in range(0,30):
		tempx,tempy = sortByFeature(x,y,j)
		for i in range(1,len(y)):
			if(tempy[i] == tempy[i-1]):						#only need to compute information gain when class label changes
				continue
			ch1 = np.count_nonzero(tempy[:len(tempy)-i] == 1)
			ch2 = np.count_nonzero(tempy[i:] == 1)
			ch1Entrop = entrop(ch1,len(tempy[:len(tempy)-i]))
			ch2Entrop = entrop(ch2,len(tempy[i:]))
			gain = float(pEntrop - float((float(len(tempy[:len(tempy)-i]))/len(y)))*ch1Entrop) + (float(float(len(tempy[i:]))/len(y))*ch2Entrop)
			
			if gain > maxGain[0]:
				maxGain = [gain,i,j]
	tempx,tempy = sortByFeature(x,y,maxGain[2])
	threshold = tempx[maxGain[1],maxGain[2]]
	#print threshold
	x,y = loadCSV("knn_test.csv")
	x,y = sortByFeature(x,y,maxGain[2])
	good = 0
	bad = 0
	for i in range(0,len(y)):
		if (float(x[i,maxGain[2]]) >= float(threshold)) and (float(y[i]) == 1): 
			good+= 1
		elif (float(x[i,maxGain[2]]) <= float(threshold)) and (float(y[i]) != 1): 
			good+= 1
		else:
			bad+= 1
	print "accuracy on testing data: "
	print float(good/float(good+bad))
	
	x,y = loadCSV("knn_train.csv")
	x,y = sortByFeature(x,y,maxGain[2])
	good = 0
	bad = 0
	for i in range(0,len(y)):
		if (float(x[i,maxGain[2]]) >= float(threshold)) and (float(y[i]) == 1): 
			good+= 1
		elif (float(x[i,maxGain[2]]) <= float(threshold)) and (float(y[i]) != 1): 
			good+= 1
		else:
			bad+= 1
	print "accuracy on training data: "
	print float(good/float(good+bad))
DT()
'''
print "PRE SORTED:"
for i in range(0,len(y)):
	print x[i,0], y[i,0]
tempx,tempy = sortByFeature(x,y,0)
print "SORTED"
for i in range(0,len(y)):
	print tempx[i,0], tempy[i,0]
	'''