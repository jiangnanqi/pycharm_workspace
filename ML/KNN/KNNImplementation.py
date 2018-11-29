import csv
import random
import operator
import numpy as np
from sklearn import datasets

def loadDataSet(filename,split,trainingSet = [],testSet = []):
    with open(filename,'rb') as  csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
                if random.random() <split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])

def euclideanDistance(instance1,instance2):
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    distance = np.sqrt(np.sum((instance1-instance2)**2))
    return distance

def getNeighbors(trainingSet,testInstance,k):
    trainingSet1 = np.array(trainingSet)[:,:-1]
    testInstance1 = np.array(testInstance)
    distance = []
    for train in trainingSet1:
        distance.append([euclideanDistance(train,testInstance),train])
    distance.sort(key=lambda d: d[0])
    neithbors = []
    for neithbor in distance[:k]:
        neithbors.append(neithbor[1])
    return neithbors

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffmat = np.tile(inX,(dataSetSize,1))-dataSet
    sqdiffmat = diffmat**2
    sqdistance = sqdiffmat.sum(axis=1)
    distance = sqdistance**0.5
    print(distance)
    sortedDistIndicies = np.argsort(distance)
    print(sortedDistIndicies)
    classCount = {}
    for i in range(k):
        votelabel = labels[sortedDistIndicies[i]]
        print(sortedDistIndicies[i])
        if votelabel in classCount:
            classCount[votelabel] = classCount.get(votelabel) + 1
        else:
            classCount[votelabel] = 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


list1 = np.array([[1,2],[2,2],[1,1],[2,1.5]])
label = np.array([3,2,2,2])
list2 = np.array([0,0])
print(classify0(list2,list1,label,3))

data = datasets.load_iris()
# print(data.data)
# print(data.target)
# test = np.array([])
# result = classify0([6.0,3.0,5.0,1.5],data.data,data.target,5)
# print(result)
