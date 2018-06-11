import math
import numpy as np
import csv

def loadCSV(dataFile, indexFile):
	keys = []
	features = []
	castData = []
	indiceso = []
	with open(indexFile, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		indices = list(lines)
	with open(dataFile, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
	for id in indices:
		indiceso.append(int(id[0]))
	for i in range(len(dataset)):
		castData.append([])
		for j in range(len(dataset[0])):
			if j != 0:
				castData[i].append(float(dataset[i][j]))
			else:
				castData[i].append(timeParse(dataset[i][j]))
	for x in range(len(castData)):
		keys.append(castData[x][-1])
		features.append(castData[x][:-1])
	return keys,features,np.array(indiceso)
	
def loadTestData(dataFile):
	castData = []
	with open(dataFile, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
	for i in range(len(dataset)):
		castData.append([])
		for j in range(len(dataset[0])):
			castData[i].append(float(dataset[i][j]))
	return castData

def compileHalfHour(keys,features,indices):
	compiledData = []
	valueCount = 0.0
	lasttime = indices[-1]
	halfHours = (lasttime//7)+1
	for x in range(halfHours):
		compiledData.append([0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0])
		valueCount = 0
		idx = 0
		for y in features:
			if (indices[idx]//7) == x:
				valueCount += 1.0
				compiledData[x][0] += y[0]
				compiledData[x][1] += y[1]
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
				compiledData[x][9] += keys[idx]
			idx += 1
		if valueCount > 0:
			compiledData[x][0] = compiledData[x][0]/valueCount
			compiledData[x][1] = compiledData[x][1]/valueCount
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
		if compiledData[x][-1] > 0:
			compiledData[x][-1] = 1
	Y = []
	X = []
	for entry in compiledData:
		Y.append(entry[-1])
		X.append(entry[0:-1])
	Xout = np.array(X)
	Yout = np.array(Y)
	return Xout,Yout

def concatHalfHour(keys,features,indices):
	concatData = []
	emptyArray = [0.0]*64
	hypoevent = 0.0
	lasttime = indices[-1]
	halfHours = (lasttime//7)+1
	for x in range(halfHours):
		concatData.append(emptyArray)
		valueCount = 0.0
		idx = 0
		concatentry = 0
		for elem in features:
			if (indices[idx]//7) == x:
				for cin in range(concatentry,7):
					concatData[x][cin] = elem[0]
					concatData[x][cin+7] = elem[1]
					concatData[x][cin+14] = elem[2]
					concatData[x][cin+21] = elem[3]
					concatData[x][cin+28] = elem[4]
					concatData[x][cin+35] = elem[5]
					concatData[x][cin+42] = elem[6]
					concatData[x][cin+49] = elem[7]
					concatData[x][cin+56] = elem[8]
				if concatData[x][63] < 1.0:
					concatData[x][63] += keys[idx]
			concatentry = indices[idx]%7
			idx += 1	
	Y = []
	X = []
	for entry in concatData:
		Y.append(entry[-1])
		X.append(entry[0:-1])
	Xout = np.array(X)
	Yout = np.array(Y)
	return Xout,Yout

def timeParse(timestring):
	time = timestring[11:-1]
	hour,min,sec = time.split(":")
	houri = int(hour)
	return houri

def compileTestdata(dataFile):
	keys = []
	features = []
	castData = []
	indiceso = []
	with open(dataFile, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
	for i in range(len(dataset)):
		castData.append([])
		for j in range(len(dataset[0])):
			castData[i].append(float(dataset[i][j]))
	newData = np.array(castData)
	compiledData = []
	valueCount = 0.0
	lasttime = len(newData)
	halfHours = (lasttime//7)+1
	for x in range(halfHours):
		compiledData.append([0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0])
		for cin in range(7):
			compiledData[x][0] += newData[x][cin]
			compiledData[x][1] += newData[x][cin+7]
			compiledData[x][2] += newData[x][cin+14]
			compiledData[x][3] += newData[x][cin+21]
			compiledData[x][4] += newData[x][cin+28]
			compiledData[x][5] += newData[x][cin+35]
			compiledData[x][6] += newData[x][cin+42]
			compiledData[x][7] += newData[x][cin+49]
			compiledData[x][8] += newData[x][cin+56]
		compiledData[x][0] = compiledData[x][0]/7.0
		compiledData[x][1] = compiledData[x][1]/7.0
		compiledData[x][2] = compiledData[x][2]/7.0
		compiledData[x][3] = compiledData[x][3]/7.0
		compiledData[x][4] = compiledData[x][4]/7.0
		compiledData[x][5] = compiledData[x][5]/7.0
		compiledData[x][6] = compiledData[x][6]/7.0
		compiledData[x][7] = compiledData[x][7]/7.0
		compiledData[x][8] = compiledData[x][8]/7.0
	Y = []
	X = []
	for entry in compiledData:
		Y.append(entry[-1])
		X.append(entry[0:-1])
	Xout = np.array(X)
	Yout = np.array(Y)
	return Xout,Yout


oy,ox,ids = loadCSV("data/Subject_1.csv","data/list_1.csv")
conx,cony = concatHalfHour(oy,ox,ids)
comx,comy = compileHalfHour(oy,ox,ids)

print conx[1]
print conx[1].size
print conx.shape

print comx[0]
print comx[0].size
print comx.shape