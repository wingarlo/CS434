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
    group=-1
    prob=-1
    for j in distance:
        if(j[1]==0):
            zero+=1
        elif(j[1]==1):
            one+=1

    if(zero>one):
        group=0
        prob=float(zero)/(zero+one)
    else:
        group=1
        prob=float(one)/(zero+one)

    return group,prob

def knn(X1,Y1,X2,Y2, predictionFile):
    n1=len(Y1)
    n2=len(Y2)

    TP=0.
    FN=0.
    FP=0.
    TN=0.

    file = open(predictionFile,"w")

    k=11
    #TESTING
    TestErr=0
    for i in range(0,n2):
        group,prob=classify(X1,Y1,X2[i],k)
        file.write(str(prob)+","+str(group)+"\n")
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

    file.close()

    print("TP: "+str(TP))
    print("FN: "+str(FN))
    print("FP: "+str(FP))
    print("TN: "+str(TN))
    # precision = TP/(TP+FP)
    # recall = TP/(TP+FN)
    # F = 2*recall*precision/(recall+precision)
    # print("precision: "+str(precision))
    # print("recall: "+str(recall))
    # print("F: "+str(F))

    return TestErr










###SUBJECT 2 INDIVIDUAL###
#Training data
oy,ox,ids = loadingdata.loadCSV("data/Subject_2_part1.csv","data/list2_part1.csv")
comx,comy = loadingdata.compileHalfHour(oy,ox,ids)
Sub2X1 = np.array(comx)
Sub2Y1 = np.array(comy)

#testing data
Sub2X2,Sub2Y2 = loadingdata.compileTestdata("data/subject2_instances.csv")

print("Subject 2")
result=knn(Sub2X1,Sub2Y1,Sub2X2,Sub2Y2,"predictions/individual1_pred1.csv") 
print("Error: "+str(result)+"\n")


###SUBJECT 7 INDIVIDUAL###
#Training data
oy,ox,ids = loadingdata.loadCSV("data/Subject_7_part1.csv","data/list_7_part1.csv")
comx,comy = loadingdata.compileHalfHour(oy,ox,ids)
Sub7X1 = np.array(comx)
Sub7Y1 = np.array(comy)

#testing data
Sub7X2,Sub7Y2 = loadingdata.compileTestdata("data/subject7_instances.csv")

print("Subject 7")
result=knn(Sub7X1,Sub7Y1,Sub7X2,Sub7Y2,"predictions/individual2_pred1.csv") 
print("Error: "+str(result)+"\n")


###GENERAL###
#Training data
oy,ox,ids = loadingdata.loadCSV("data/Subject_1.csv","data/list_1.csv")
comx,comy = loadingdata.compileHalfHour(oy,ox,ids)
Sub1X1 = np.array(comx)
Sub1Y1 = np.array(comy)

oy,ox,ids = loadingdata.loadCSV("data/Subject_4.csv","data/list_4.csv")
comx,comy = loadingdata.compileHalfHour(oy,ox,ids)
Sub4X1 = np.array(comx)
Sub4Y1 = np.array(comy)

oy,ox,ids = loadingdata.loadCSV("data/Subject_6.csv","data/list_6.csv")
comx,comy = loadingdata.compileHalfHour(oy,ox,ids)
Sub6X1 = np.array(comx)
Sub6Y1 = np.array(comy)

oy,ox,ids = loadingdata.loadCSV("data/Subject_9.csv","data/list_9.csv")
comx,comy = loadingdata.compileHalfHour(oy,ox,ids)
Sub9X1 = np.array(comx)
Sub9Y1 = np.array(comy)

#combine training data into one numpy matrix
GenX1 = np.concatenate((Sub1X1, Sub4X1, Sub6X1, Sub9X1), axis=0)
GenY1 = np.concatenate((Sub1Y1, Sub4Y1, Sub6Y1, Sub9Y1))

#Testing data
GenX2,GenY2 = loadingdata.compileTestdata("data/general_test_instances.csv")

print("General")
result=knn(GenX1,GenY1,GenX2,GenY2,"predictions/general_pred1.csv") 
print("Error: "+str(result)+"\n")