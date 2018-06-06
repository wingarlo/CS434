import csv
import numpy as np

def sigm(x):
	return (1 / (1 + np.exp(-x)))
def loadCSV(fileName):
	data = []
	xt = []
	yt = []
	x = []
	y = []
	with open(fileName, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)									#lines = list of every data point
	dataset = np.array(dataset)
	
	for i in range (0, len(dataset)):
		data.append(np.delete(dataset[i],[0]))
	for i in range(0,len(data)):
		for j in range(0,len(data[i])):
			data[i][j] = sigm(float(data[i][j]))
	data = np.array(data)
	for i in range (0, len(dataset)):
		xt.append(data[i][:-1])
		yt.append([data[i][8]])
	xt = np.array(xt)
	yt = np.array(yt)
	xt = xt.astype(np.float)
	yt = yt.astype(np.float)
	for i in range(0,len(yt)-14,7):
		x.append(np.concatenate([xt[i],xt[i+1],xt[i+2],xt[i+3],xt[i+4],xt[i+5],xt[i+6]]))
		y.append(yt[i]+yt[i+1]+yt[i+2]+yt[i+3]+yt[i+4]+yt[i+5]+yt[i+6])
	x = np.array(x)
	y = np.array(y)
	for i in range(0,len(y)):
		if (y[i][0] >= 1):
			y[i][0] = 1
	x1 = np.matrix(x.astype(np.float))
	y1 = np.matrix(y.astype(np.int))


	return x1,y1

x,y = loadCSV('./data/Subject_2_part1.csv')		#load training data
w = np.matrix([])
np.savetxt("foo.csv", x, delimiter=",")
w = (x.T * x).I * x.T * y					#weight = (x^T * x)^-1 * x^T * y
x,y = loadCSV('./data/Subject_2_part1.csv')		#load testing data
error = 0
'''

good = 0
bad = 0
pospos = 0
negneg = 0
falsepos = 0
falseneg = 0
for i in range (len(y)):
	print "X", i, "*w",x[i]*w
	if(y[i] == [1] and x[i]*w > .5):
		good += 1
		pospos += 1
	elif(x[i]*w < .5 and y[i] == [0]):
		good += 1
		negneg += 1
	elif(x[i]*w > .5 and y[i] == [0]):
		bad += 1
		falsepos += 1
	else:
		bad += 1
		falseneg += 1
	#error += float((y[i] - x[i]*w))			#calculating error
acc = float(float(good)/float(good+bad))
print "Accuracy:"
print acc
print "Correctly predicted Positive:"
print pospos
print "Correctly predicted Negative:"
print negneg
print "false positives:"
print falsepos
print "False negatives:"
print falseneg
'''
for i in range (len(y)):
	error += float((y[i] - x[i]*w))
error = error / len(y)
print "Average error with dummy variable and added features:"
print error