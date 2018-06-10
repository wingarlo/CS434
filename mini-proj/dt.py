import numpy as np
import copy
import math
import loadingdata

class Node:
    depth=-1
    Set=None #holds subset
    pos=0 #how many 1 class elements in subset
    neg=0 #how many 0
    test=-1 #index of feature that will be the test to split this subset
    thresh=-1 #threshold index for test
    threshval=-1 #threshold value
    parent=None
    left=None #left child
    right=None
    entropy=0

def dt(X1,X2):
    csvData=[]
    d=1
    while(d<3):
        tests=[]
        root=buildTree(X1,d,tests)
        trainError=0
        testError=0

        #training
        for example in X1:
            predict=classifyTree(root,example)
            if(predict!=example[len(example)-1]):
                trainError+=1
        #trainError=float(trainError)/len(X1)


        # #testing
        # for example in X2:
        #     predict=classifyTree(root,example)
        #     if(predict!=example[0]):
        #         testError+=1
        # testError=float(testError)/len(X2)


        # csvData.append(str(d)+","+str(trainError)+","+str(testError))
        print("d: "+str(d)+" Training Error: "+str(trainError)+" Testing Error: "+str(testError))
        printTree(root,d)

        d+=1
    # print("Graph of results can be found in the write-up")
    # with open('part2.csv','wb') as file:
    #     for line in csvData:
    #         file.write(line)
    #         file.write('\n')

    
    
    return 0

def printTree(root,levels):
    print("Level "+str(levels-root.depth)+" neg: "+str(root.neg)+" pos: "+str(root.pos)+" test: "+str(root.test)+" threshval: "+str(root.threshval)+" entropy: "+str(root.entropy))
    if(root.left!=None):
        printTree(root.left,levels)
    if(root.right!=None):
        printTree(root.right,levels)


def buildTree(X,d,tests): #takes set, depth, and tests, returns an expanded node
    root=Node()
    root.Set=X
    counts=count(root.Set)
    root.neg=counts[0]
    root.pos=counts[1]
    root.entropy=calcEntropy(root.neg,root.pos,len(root.Set))
    root.depth=d

    if(d>0):
        if(root.entropy>0):
            #find best test to split on
            bestTest(root,tests)
            #sort set based on test feature
            root.Set=sorted(root.Set,key=lambda x: x[root.test])
            root.threshval=root.Set[root.thresh][root.test]
            #split set into left and right
            Left=root.Set[:root.thresh]
            Right=root.Set[root.thresh:]
            #add test to used tests
            tests.append(root.test)
            #expand children
            left=buildTree(Left,d-1,tests)
            right=buildTree(Right,d-1,tests)
            
            root.left=left
            root.right=right


    return root

def bestTest(node,tests): #finds index of feature test with most gain. updates node's test and thresh index after sorting
    maxGain=1.0
    maxIndex=-1
    thresh=-1

    for i in range(0,10): #for each feature
        if(contains(tests,i)==False):
            result=bestThresh(node,i) #returns (gain, threshold index)
            if(result[0]<maxGain):
                maxGain=result[0]
                maxIndex=i
                thresh=result[1]
    node.test=maxIndex
    node.thresh=thresh

def bestThresh(node,f): #returns (gain, threshold index)
    maxGain=1.0
    thresh=-1

    examples=sorted(node.Set,key=lambda x: x[f])
    i=1
    while(i<len(examples)):
        if(examples[i][f]!=examples[i-1][f]): #class changed, calculate gain at this split
           
            Left,Right=np.split(examples,[i])
            lsplit=count(Left) #tuple (number of -1,number of 1)
            rsplit=count(Right)

            HL=calcEntropy(lsplit[0],lsplit[1],len(Left))
            HR=calcEntropy(rsplit[0],rsplit[1],len(Right))
            gain=HL+HR

            if(gain<maxGain):
                maxGain=gain
                thresh=i
        i+=1
    return (maxGain,thresh)

def contains(list,item):
    for elt in list:
        if(elt==item):
            return True
    return False



def count(Set):
    neg=0
    pos=0
    for elt in Set:
        if(elt[0]==0):
            neg+=1
        else:
            pos+=1
    return (neg,pos)

def calcEntropy(neg,pos,length): #entropy of individual subset, not total tree
    p1=float(neg)/length
    p2=float(pos)/length
    if(p1==0 and p2==0):
        return 0
    elif(p1==0):
        return -p2*math.log(p2,2)
    elif(p2==0):
        return -p1*math.log(p1,2)
    else:
        return -p1*math.log(p1,2)-p2*math.log(p2,2)

def classifyTree(root,example):
    current=root
    while(current!=None):
        feat=current.test
        threshval=current.threshval

        if(feat!=-1):
            if(example[feat]<threshval):
                current=current.left
            else:
                current=current.right
        elif(current.entropy==0):
            if(current.pos>0):
                return 1
            else:
                return 0
        else:
            if(current.pos>current.neg):
                return 1
            else:
                return 0
        
    return -1

okeys,ofeatures,ocastData = loadingdata.loadCSV("data/Subject_1.csv","data/list_1.csv")
Xres,Yres = loadingdata.compileHalfHour(okeys,ofeatures,ocastData)
X = np.array(Xres)
Y = np.array(Yres)

X=np.concatenate((Y[:,None],X),axis=1)
print(X)
dt(X,None)