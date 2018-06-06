import numpy as np
import copy
import math
import loadingdata

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
    n2=len(Y2)

    TP=0.
    FN=0.
    FP=0.
    TN=0.

    k=2
    #TESTING
    TestErr=0
    for i in range(0,n2):
        group=classify(X1,Y1,X2[i],k)
        if(group!=Y2[i]):
            TestErr+=1
            if(group==1):
                FP+=1
            else:
                FN+=1
        else:
            if(group==1):
                TP+=1
            else:
                TN+=1
    TestErr=float(TestErr)/n2 #percentage of wrong predictions

    print("TP: "+str(TP))
    print("FN: "+str(FN))
    print("FP: "+str(FP))
    # precision = TP/(TP+FP)
    # recall = TP/(TP+FN)
    # F = 2*recall*precision/(recall+precision)

    return TestErr




okeys,ofeatures,ocastData = loadingdata.loadCSV("data/Subject_1.csv","data/list_1.csv")
Xres,Yres = loadingdata.compileHalfHour(okeys,ofeatures,ocastData)
X = np.array(Xres)
Y = np.array(Yres)

result=knn(X,Y,X,Y) #training err
print(result)