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
		castData[i][0] = int(indices[i][0])
	for x in range(len(castData)):
		keys.append(castData[x][-1])
		features.append(castData[x][:-1])
	return keys,features,castData
	
def compileHalfHour(keys,features,castData):
	compiledData = []
	valueCount = 0.0
	halfHours = (len(keys)//7)+1
	for x in range(halfHours):
		compiledData.append([0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0])
		valueCount = 0
		for y in castData:
			if (y[0]//7) == x:
				valueCount += 1.0
				compiledData[x][1] += y[1]
				compiledData[x][2] += y[2]
				compiledData[x][3] += y[3]
				compiledData[x][4] += y[4]
				compiledData[x][5] += y[5]
				compiledData[x][6] += y[6]
				compiledData[x][7] += y[7]
				compiledData[x][8] += y[8]
		if valueCount > 0:
			compiledData[x][0] = x
			compiledData[x][1] = compiledData[x][1]/valueCount
			compiledData[x][2] = compiledData[x][2]/valueCount
			compiledData[x][3] = compiledData[x][3]/valueCount
			compiledData[x][4] = compiledData[x][4]/valueCount
			compiledData[x][5] = compiledData[x][5]/valueCount
			compiledData[x][6] = compiledData[x][6]/valueCount
			compiledData[x][7] = compiledData[x][7]/valueCount
			compiledData[x][8] = compiledData[x][8]/valueCount
		else:
			compiledData[x][0] = -1
		compiledData[x].insert(-1,valueCount)
		if compiledData[x][-1] > 0:
			compiledData[x][-1] = 1
	return compiledData
	
okeys,ofeatures,ocastData = loadCSV("data/Subject_1.csv","data/list_1.csv")
results = np.array(compileHalfHour(okeys,ofeatures,ocastData))
print results.shape
print results

				
				