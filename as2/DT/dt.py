import numpy as np
import math
import sys
import csv
gains = []
thresholds = []
tree = []
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
	print "accuracy on testing data using decision stump: "
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
	print "accuracy on training data using decision stump: "
	print float(good/float(good+bad))
	
def recurDT(x,y,d):
	maxGain = [0,0,0]
	if d!=0:
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
		
		thresholds.append(float(tempx[maxGain[1], maxGain[2]]))
		gains.append(maxGain[2])
		newx = x
		x = x[x[:,maxGain[2]].argsort()]
		
		for i in range(0,len(x)):
			for j in range(0,len(x[i,0])):
				newx[i,j] = x[i,0,j]
		x = newx
		x = x[:len(x)-maxGain[1]]
		y = y[:len(y)-maxGain[1]]
		dummy, y = sortByFeature(x,y,maxGain[2])
		tree.append(recurDT(x,y,d-1))
	
	

	return x
	
def DeepTree(d):
	x,y = loadCSV("knn_train.csv")
	tree.append(recurDT(x,y,d))
	x,y = loadCSV("knn_test.csv")
	good = 0
	bad = 0
	for q in range(0,d):
		newx = x
		x = x[x[:,gains[q]].argsort()]
		
		for i in range(0,len(x)):
			for j in range(0,len(x[i,0])):
				newx[i,j] = x[i,0,j]
		x = newx
	

		for i in range(0,len(y)):
			if (float(x[i,gains[q]]) >= float(thresholds[q])) and (float(y[i]) == 1): 
				good+= 1
			elif (float(x[i,gains[q]]) <= float(thresholds[q])) and (float(y[i]) != 1): 
				good+= 1
			else:
				bad+= 1
	print "accuracy on testing data using Decision tree of depth", d
	print float(good/float(good+bad))

	x,y = loadCSV("knn_train.csv")
	good = 0
	bad = 0
	for q in range(0,d):
		newx = x
		x = x[x[:,gains[q]].argsort()]
		
		for i in range(0,len(x)):
			for j in range(0,len(x[i,0])):
				newx[i,j] = x[i,0,j]
		x = newx
	

		for i in range(0,len(y)):
			if (float(x[i,gains[q]]) >= float(thresholds[q])) and (float(y[i]) == 1): 
				good+= 1
			elif (float(x[i,gains[q]]) <= float(thresholds[q])) and (float(y[i]) != 1): 
				good+= 1
			else:
				bad+= 1
	print "accuracy on training data using Decision tree of depth", d
	print float(good/float(good+bad))
	
d = 5
if len(sys.argv) > 1:
	d = int(sys.argv[1])
DT()
DeepTree(d)
