import numpy as np
import copy
import math
import loadingdata

# def readData(inputFile,indices,featuresMatrix,labelsMatrix):
#     #read in training data
#     file1 = open(inputFile,"r")
#     file2 = open(indices,"r")
#     for line,index in zip(file1,file2):
#         #put values into matrices
#         row = line.split(',') #array of elements in current row
#         labelsMatrix.append(row.pop()) #append class
#         row[0]=index
#         featuresMatrix.append(row) #add the rest of elements in feature matrix
#     file1.close()
#     file2.close()

# A = []
# Alabels = []

# readData("data/Subject_1.csv","data/list_1.csv",A,Alabels)
# A = np.asarray(A,dtype=float)
# Alabels = np.asarray(Alabels,dtype=float)

def classify(X, Y, p, k): #classifies one point. p=test point
    distance=[]
    for i in range(0,len(X)):
        d=np.sqrt(np.dot((p-X[i]),(p-X[i])))
        distance.append((d,Y[i])) #tuple: distance, class
    distance=sorted(distance)[:k]

    zero=0
    one=0
    for j in distance:
        if(j[1]==0):
            zero+=1
        elif(j[1]==1):
            one+=1

    return 0 if zero>one else 1

def knn(X1,Y1,X2,Y2):
    n1=len(Y1)
    #n2=len(Y2)

    TP=0.
    FN=0.
    FP=0.
    TN=0.

    k=2
    #TRAINING ERROR
    TrainErr=0
    for i in range(0,n1):
        group=classify(X1,Y1,X1[i],k)
        if(group!=Y1[i]):
            TrainErr+=1
            if(group==1):
                FP+=1
            else:
                FN+=1
        else:
            if(group==1):
                TP+=1
            else:
                TN+=1
    TrainErr=float(TrainErr)/n1 #percentage of wrong predictions

    print("TP: "+str(TP))
    print("FN: "+str(FN))
    print("FP: "+str(FP))
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F = 2*recall*precision/(recall+precision)

    return precision




okeys,ofeatures,ocastData = loadingdata.loadCSV("data/Subject_1.csv","data/list_1.csv")
Xres,Yres = loadingdata.compileHalfHour(okeys,ofeatures,ocastData)
X = np.array(Xres)
Y = np.array(Yres)

result=knn(X,Y,None,None)
print(result)