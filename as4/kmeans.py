import sys
import numpy as np
import os
import math
from random import randint

k = 2
if len(sys.argv) == 2:
	k = int(sys.argv[1])

	
def dist(p1, p2):#takes in 2 arrays of any length and returns distance
	distance = 0
	for i in range(0,len(p1)):
		distance += float((float(p1[i])-float(p2[i]))**2)
	return math.sqrt(distance)

def readData(filename):
	x = []
	with open(filename) as f:
		for lines in f:
			lines = lines[:-1]#getting rid of \n
			x.append(lines.split(','))
	x = np.array(x)
	return x.astype(np.int)#convert from string to int

	
data = readData("./data/data-1.txt")
mu = []
c = []
for i in range(k):#initialization
	mu.append(data[randint(0,len(data))])
	c.append([])
temp = 0
for count in range(0,10):
	for x in range(len(data)):
		minDist = 99999
		for i in range(k):#assignment step
			if (dist(data[x],mu[i]) < minDist):
				minDist = dist(data[x],mu[i])
				
				temp = i			#finds which mu data[x] is closest to, and puts it in c[i]
		c[temp].append(data[x])
	
	for j in range(len(c)):			#Update step
		mu[j] = []
		for x in range(0,len(data[0])):
			sum = 0
			for y in range(0,len(data)):
				sum += data[y,x]
			mu[j].append((sum/len(data)))
	
	print len(c[0])
	print len(c[1])
	for j in range(0,len(c)):
		c[j] = []
	
		
