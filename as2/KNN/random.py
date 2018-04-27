import numpy as np
import math
import sys
import csv

#Functions
def loadCSV(fileName):
	with open(fileName, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)									#lines = list of every data point
	print dataset
	

def get_Dist(example1, example2, length):
	distance = 0.0
	for i in range(length):
		distance += (example1[i] - example2[i])**2
	return math.sqrt(distance)
loadCSV("knn_train.csv")