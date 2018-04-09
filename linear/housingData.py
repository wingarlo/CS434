import numpy as np
temp = []
x = []
y = []
with open('./housing_train.txt') as f:
	lines = f.read().split()
for count in range (len(lines)):
	if (count+1) % 14 == 0:
		y.append([float(lines[count])])
	else:
		if (count+1) % 14 == 1:
			temp.append(1.)
		temp.append(float(lines[count]))
x = [temp[i:i+14] for i in range(0,len(temp),14)]
x1 = np.matrix(x)
y1 = np.matrix(y)
w = np.matrix([])
w = (x1.T * x1).I * x1.T * y1
#print w


with open('./housing_test.txt') as f:
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
error = []
for i in range (len(y1)):
	error.append(y1[i] - x1[i]*w)
print error