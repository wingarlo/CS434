import numpy as np
import csv
e = 2.71828182846
numEx = 1400      #number of examples
emptyArray = []


x = []
y = []
temp = []
with open("usps-4-9-train.csv") as f:
	string = f.read().replace('\n',',')
	lines = string.split(',')			#lines = list of every data point
lines = lines[:-1]
for count in range (0,len(lines)):
	if (count+1) % 257 == 0:	#if position is 14(y value) append to y
		y.append([float(lines[count])])
	else:								#else (if first position append a dummy '1') append to temp
		temp.append(float(lines[count])/(255.*2.))
x = [temp[i:i+256] for i in range(0,len(temp),256)]		#splitting x into a list of lists separated by each instance
x1 = np.matrix(x)						#converting x and y from lists of lists to matrices
y1 = np.matrix(y)

for f in range(256):
	emptyArray.append(0)
w = np.matrix(emptyArray)
gradient = np.matrix(emptyArray)
eta = .01 #learning rate
epsilon = 10 #target gradient
cont = True
itr =0
while (cont):
	gradient = np.matrix(emptyArray)
	for i in range(0,(numEx)):
		power = float(-1*(w * x1[i].T))
		ytarget = 1/(1+(e**power))
		gradient = gradient + ((ytarget - y1[i]))* x1[i]
	w = w - (eta*gradient)
	itr +=1
	cont = (itr < 100)

x = []
y = []
temp = []
	
with open("usps-4-9-test.csv") as f:
	string = f.read().replace('\n',',')
	lines = string.split(',')			#lines = list of every data point
lines = lines[:-1]
for count in range (0,len(lines)):
	if (count+1) % 257 == 0:	#if position is 14(y value) append to y
		y.append([float(lines[count])])
	else:								#else (if first position append a dummy '1') append to temp
		temp.append(float(lines[count])/(255.*2.))
x = [temp[i:i+256] for i in range(0,len(temp),256)]		#splitting x into a list of lists separated by each instance
x1 = np.matrix(x)						#converting x and y from lists of lists to matrices
y1 = np.matrix(y)
good = 0
bad = 0
for i in range (0,800):
	power = float(-1*(w * x1[i].T))
	error = float(y1[i] - (1/(1+e**power)))
	print "Error = " , error
	if abs(error) >= 0.5:
		bad += 1
	else:
		good += 1

percent = float(good)/float(good+bad)
print "percent accuracy: " , percent*100.
print "Number of errors: " , bad

	