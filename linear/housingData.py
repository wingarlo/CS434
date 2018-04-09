import numpy as np

def adFeature(filename, d):
	
	x = []
	y = []
	temp = []
	phil = 0;
	with open(filename) as f:
		lines = f.read().split()
	xd = np.random.normal(0,0.1,(len(lines)/(14))*d)
	for count in range (0,len(lines)):
		if (count+1) % 14 == 0:
			y.append([float(lines[count])])
			for i in range(0,d):
				temp.append(float(xd[phil]))
				phil+=1
		else:
			if (count+1) % 14 == 1:
				temp.append(1.)
			temp.append(float(lines[count]))
	x = [temp[i:i+14+d] for i in range(0,len(temp),14+d)]
	x1 = np.matrix(x)
	y1 = np.matrix(y)
	return x1,y1
#loadData w/out dummy var
def loadData2(filename):
	x = []
	y = []
	temp = []
	with open(filename) as f:
		lines = f.read().split()
	for count in range (0,len(lines)):
		if (count+1) % 14 == 0:
			y.append([float(lines[count])])
		else:
			temp.append(float(lines[count]))
	x = [temp[i:i+13] for i in range(0,len(temp),13)]
	x1 = np.matrix(x)
	y1 = np.matrix(y)
	return x1,y1

#loadData with dummy var
def loadData(filename):
	x = []
	y = []
	temp = []
	with open(filename) as f:
		lines = f.read().split()
	for count in range (0,len(lines)):
		if (count+1) % 14 == 0:
			y.append([float(lines[count])])
		else:
			if (count+1) % 14 == 1:
				temp.append(1.)
			temp.append(float(lines[count]))
	x = [temp[i:i+14] for i in range(0,len(temp),14)]
	x1 = np.matrix(x)
	y1 = np.matrix(y)
	return x1,y1


x,y = loadData('./housing_train.txt')		#load training data
w = np.matrix([])
w = (x.T * x).I * x.T * y					#weight = (x^T * x)^-1 * x^T * y


x,y = loadData('./housing_test.txt')		#load testing data
error = 0
for i in range (len(y)):
	error += float((y[i] - x[i]*w))			#calculating error
error = error / len(y)						#average
print "Average error with dummy variable:"
print error

##################################################
#Again without dummy variable
##################################################

x,y = loadData2('./housing_train.txt')
w = np.matrix([])
w = (x.T * x).I * x.T * y
#print w


x,y = loadData2('./housing_test.txt')
error = 0
for i in range (len(y)):
	error += float((y[i] - x[i]*w))
error = error / len(y)
print "Average error without dummy variable:"
print error

##################################################
#Again withdummy variable and added features
##################################################

x,y = adFeature('./housing_train.txt',2)
w = np.matrix([])
w = (x.T * x).I * x.T * y
#print w


x,y = adFeature('./housing_test.txt',2)
error = 0
for i in range (len(y)):
	error += float((y[i] - x[i]*w))
error = error / len(y)
print "Average error with dummy variable and added features:"
print error