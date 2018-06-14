import math
import numpy as np
import csv
import sys
import loadingdata as loadD

e = 2.718281828459045235360287471352662497757247093699959574966967627724076630353547594571382178525166427427466391932003059921817413596629043572900334295260595630738132328627943490763233829880753195251019011573834187930702154089149934884167509244761460668082264800168477411853742345442437107539077744992069551702761838606261331384583000752044933826560297606737113200709328709127443747047230696977209310141692836819025515108657463772111252389784425056953696770785449969967946864454905987931636889230098793127736178215424999229576351482208269895193668033182528869398496465105820939239829488793320362509443117301238197068416140397019837679320683282376464804295311802328782509819455815301756717361332069811250996181881593041690351598888519345807273866738589422879228499892086805825749279610484198444363463244968487560233624827041978623209002160990235304369941849146314093431738143640546253152096183690888707016768396424378140592714563549061303107208510383750510115747704171898610687396965521267154688957035035402123407849819334321068170121005627880235193033224745015853904730419957777093503660416997329725088687696640355570716226844716256079882651787134195124665201030592123667719432527867539855894489697096409754591856956380236370162112047742722836489613422516445078182442352948636372141740238893441247963574370263755294448337998016125492278509257782562092622648326277933386566481627725164019105900491644998289315056604725802778631864155195653244258698294695930801915298721172556347546396447910145904090586298496791287406870504895858671747985466775757320568128845920541334053922000113786300945560688166740016984205580403363795376452030402432256613527836951177883863874439662532249850654995886234281899707733276171783928034946501434558897071942586398772755
maxpow = (math.log(sys.float_info.max))-0.1 #maximum power that e can be put to and not overflow
minpow = (math.log(sys.float_info.min))+0.1 #minimum power that e can be put to and not overflow
testFiles = ["./data/general_test_instances.csv","./data/subject2_instances.csv","./data/subject7_instances.csv"]
testOutputs = ["./predictions/general_pred3.csv","./predictions/individual1_pred3.csv","./predictions/individual2_pred3.csv"]

def LogReg(xi,yi,eta,itr):
	x = np.matrix(xi)
	y = np.matrix(yi)
	(numEx,numFeat) = x.shape
	emptyArray = [0.0]*numFeat
	w = np.matrix(emptyArray)
	gradient = np.matrix(emptyArray)
	print "Learning Rate: " +repr(eta)
	print "Iterations: " +repr(itr)
	for f in range(itr):
		gradient = np.matrix(emptyArray)
		for i in range(0,numEx):
			power = float(-1*(w * x[i].T))
			if (power >= maxpow):
				ytarget = 0.0
			elif (power <= minpow):
				ytarget = 1.0
			else:
				ytarget = 1/(1+(e**power))
			if x[i,0] != -1:
				gradient = gradient + ((ytarget - y[0,i])*x[i])
		w = w - (eta*gradient)
	return w
	
def testAccuracy(weight,feati,resulti):
	id = 0
	emptyHalfHourCount = 0
	featNum = len(feati)
	while id < featNum:
		if feati[id][0] == -1:
			feati = np.delete(feati,id,0)
			resulti = np.delete(resulti,id,0)
			featNum -= 1
			emptyHalfHourCount += 1
		id +=1

	feats = np.matrix(feati)
	results = np.matrix(resulti)
	results = results.T
	nEx = results.size
	correct = 0.0
	wrong = 0
	fpos,fneg = 0,0
	for i in range(nEx):
		powert = float(-1*(weight * feats[i].T))
		if (powert >= maxpow):
			ytest = 0.0
		elif (powert <= minpow):
			ytest = 1.0
		else:
			ytest = 1/(1+(e**powert))
			
		if (round(ytest,1) == results[i,0]):
			correct += 1.0
		else:
			wrong += 1
			if round(ytest,1) == 1.0:
				fpos += 1
			else:
				fneg += 1
				
	percentage = (float(correct/float(nEx)))*100
	percentFpos = (float(fpos/float(wrong)))*100
	percentFneg = (float(fneg/float(wrong)))*100
	print " "
	print "Percentage: " +repr(percentage)+"%"
	print "Number of errors: " +repr(wrong)
	print "False Positives: "+repr(percentFpos)+"%"
	print "False Negatives: "+repr(percentFneg)+"%"
	return percentage

def test(weight,testN):
	testData = loadD.compileTestdata(testFiles[testN])
	feats = np.matrix(testData)
	nEx = len(feats)
	results = []
	for i in range(nEx):
		powert = float(-1*(weight * feats[i].T))
		if (powert >= maxpow):
			ytest = 0.0
		elif (powert <= minpow):
			ytest = 1.0
		else:
			ytest = 1/(1+(e**powert))
		prob = np.fabs(ytest - 0.5);
		prob = prob * 2;
		results.append([prob,round(ytest,1)])
	filename = testOutputs[testN]
	with open(filename,'wb') as mlpout:
		output = csv.writer(mlpout, delimiter=',')
		for q in range(len(results)):
			output.writerow([results[q][0]]+[results[q][1]])
	return 0

okeys,ofeatures,ocastData = loadD.loadCSV("data/Subject_1.csv","data/list_1.csv")
X1,Y1 = loadD.compileHalfHour(okeys,ofeatures,ocastData)
okeys,ofeatures,ocastData = loadD.loadCSV("data/Subject_4.csv","data/list_4.csv")
X2,Y2 = loadD.compileHalfHour(okeys,ofeatures,ocastData)
okeys,ofeatures,ocastData = loadD.loadCSV("data/Subject_6.csv","data/list_6.csv")
X3,Y3 = loadD.compileHalfHour(okeys,ofeatures,ocastData)
okeys,ofeatures,ocastData = loadD.loadCSV("data/Subject_9.csv","data/list_9.csv")
X4,Y4 = loadD.compileHalfHour(okeys,ofeatures,ocastData)
X = np.concatenate((X1, X2, X3, X4), axis=0)
Y = np.concatenate((Y1, Y2, Y3, Y4), axis=0)
Dubya = LogReg(X,Y,0.01,1000)
print "Weight"
print Dubya
testAccuracy(Dubya,X1,Y1)
testAccuracy(Dubya,X2,Y2)
testAccuracy(Dubya,X3,Y3)
testAccuracy(Dubya,X4,Y4)
testAccuracy(Dubya,X,Y)
test(Dubya,0)
test(Dubya,1)
test(Dubya,2)
