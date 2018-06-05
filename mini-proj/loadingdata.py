import math
import numpy as np
import csv
import operator

def loadCSV(dataFile,indicesFile):
	keys = []
	features = []
	castData = []
	with open(dataFile, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
	with open(indicesFile, 'rb') as indfile:
		inds = csv.reader(indfile)
		indices = list(inds)
	for i in range(len(dataset)):
		castData.append([])
		for j in range(len(dataset[0])):
			if j != 0:
				castData[i].append(float(dataset[i][j]))
			else:
				castData[i].append(timeParse(dataset[i][j]))
		castData[i].insert(0,int(indices[i][0]))
	for x in range(len(castData)):
		keys.append(castData[x][-1])
		features.append(castData[x][:-1])
	return keys,features,castData
	
def compileHalfHour(keys,features,castData):
	compiledData = []
	valueCount = 0.0
	lasttime = int(castData[-1][0])
	halfHours = (lasttime//7)+1
	for x in range(halfHours):
		compiledData.append([0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0])
		valueCount = 0
		for y in castData:
			if (y[0]//7) == x:
				if valueCount == 0:
					compiledData[x][1] = y[1]
				valueCount += 1.0
				compiledData[x][2] += y[2]
				compiledData[x][3] += y[3]
				compiledData[x][4] += y[4]
				if y[5] > 0 :
					compiledData[x][5] += 1
				if y[6] > 0 :
					compiledData[x][6] += 1
				if y[7] > 0 :
					compiledData[x][7] += 1
				if y[8] > 0 :
					compiledData[x][8] += 1
				compiledData[x][9] += y[9]
		if valueCount > 0:
			compiledData[x][0] = x
			compiledData[x][2] = compiledData[x][2]/valueCount
			compiledData[x][3] = compiledData[x][3]/valueCount
			compiledData[x][4] = compiledData[x][4]/valueCount
			compiledData[x][5] = compiledData[x][5]/valueCount
			compiledData[x][6] = compiledData[x][6]/valueCount
			compiledData[x][7] = compiledData[x][7]/valueCount
			compiledData[x][8] = compiledData[x][8]/valueCount
			compiledData[x][9] = compiledData[x][9]/valueCount
		else:
			compiledData[x][0] = -1
		compiledData[x].insert(-1,valueCount)
		if compiledData[x][-1] > 0:
			compiledData[x][-1] = 1
	Y = []
	X = []
	for entry in compiledData:
		Y.append(entry[-1])
		X.append(entry[0:-1])
	return X,Y
	
def timeParse(timestring):
	time = timestring[11:-1]
	hour,min,sec = time.split(":")
	houri = int(hour)
	mini = int(min)
	seci = int(sec)
	mini = mini*60
	houri = houri*60*60
	seconds = seci + mini + houri
	return seconds

okeys,ofeatures,ocastData = loadCSV("data/Subject_1.csv","data/list_1.csv")
Xres,Yres = compileHalfHour(okeys,ofeatures,ocastData)
X = np.array(Xres)
Y = np.array(Yres)
print X.shape
print Y.shape		