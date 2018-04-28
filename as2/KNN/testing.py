import math
import csv
import operator

#Functions
def loadCSV(fileName):
	keys = []
	features = []
	castData = []
	with open(fileName, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
	for i in range(len(dataset)):
		castData.append([])
		for j in range(len(dataset[0])):
			castData[i].append(float(dataset[i][j]))
	for x in range(len(castData)):
		keys.append(castData[x][0])
		features.append(castData[x][1:])
	return keys,features,dataset

def get_Dist(example1, example2):
	distance = 0.0
	numFeatures = len(example1)
	for i in range(numFeatures):
		distance += (example1[i] - example2[i])**2
	return math.sqrt(distance)
	

def find_Neighbors(oKeys,oFeats,nKey,nFeat, k):
	distances = []
	numExamples = len(oFeats)
	for x in range(numExamples):
		dist = get_Dist(nFeat, oFeats[x])
		distances.append([oKeys[x],oFeats[x], dist])
	distances = sorted(distances, key=lambda dist: dist[2])
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x])
	return neighbors

def form_prediction(neighbors):
	voteDict = {}
	for x in range(len(neighbors)):
		response = neighbors[x][0]
		if response in voteDict:
			voteDict[response] += 1
		else:
			voteDict[response] = 1
	sortedVotes = sorted(voteDict.iteritems(), key=lambda dict: dict[1], reverse=True)
	return sortedVotes

def testAccuracy(k):
	y,x,z = loadCSV("knn_train.csv")
	yt,xt,zt = loadCSV("knn_test.csv")
	predictions = []
	correct = 0.0
	numExamples = len(zt)
	for i in range(numExamples):
		predictions.append(form_prediction(find_Neighbors(y,x,yt[i],xt[i],k))[0][0])
	for j in range(numExamples):
		if (predictions[j] == yt[j]):
			correct+=1.0
	return (correct/float(numExamples))*100.0

def main():
	y,x,z = loadCSV("knn_train.csv")
	yt,xt,zt = loadCSV("knn_test.csv")
	results = []
	predictions = []
	numExamples = len(z)
	numSamples = len(zt)
	for k in range(1,52):
		print "K: "+repr(k)
		results.append([])
		predictions = []
		correct = 0.0
		wrong = 0.0
		for i in range(numExamples):
			predictions.append(form_prediction(find_Neighbors(y,x,y[i],x[i],k))[0][0])
		for j in range(numExamples):
			if (predictions[j] == y[j]):
				correct+=1.0
			else:
				wrong+=1.0
		results[k-1].append(wrong)
		
		predictions = []
		correct = 0.0
		wrong = 0.0
		for l in range(numExamples):
			loox = []
			looy = []
			for elem in x:
				loox.append(elem)
			for elem2 in y:
				looy.append(elem2)
			del looy[l]
			del loox[l]
			predictions.append(form_prediction(find_Neighbors(looy,loox,y[l],x[l],k))[0][0])
		for m in range(numExamples):
			if (predictions[m] == y[m]):
				correct+=1.0
			else:
				wrong+=1.0
		results[k-1].append(wrong)
		
		predictions = []
		correct = 0.0
		wrong = 0.0
		for n in range(numSamples):
			predictions.append(form_prediction(find_Neighbors(y,x,yt[n],xt[n],k))[0][0])
		for o in range(numSamples):
			if (predictions[o] == yt[o]):
				correct+=1.0
			else:
				wrong+=1.0
		results[k-1].append(wrong)
	with open('KNNresults.csv','wb') as csvout:
		output = csv.writer(csvout, delimiter=',')
		for q in range(len(results)):
			output.writerow([results[q][0]]+[results[q][1]]+[results[q][2]])
	return results

print main()
