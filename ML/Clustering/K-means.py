import numpy as np
from sklearn.cluster import KMeans

def kmeans(x,k,maxIt):
    numPoints,numDim = x.shape

    dataSet = np.zeros((numPoints,numDim+1))
    dataSet[:,:-1] = x

    centroids = dataSet[np.random.randint(numPoints,size=k)]
    centroids[:,-1] = range(1,k+1)

    iteration = 0
    oldCentroids = None

    while not shouldStop(oldCentroids,centroids,iteration,maxIt):
        oldCentroids = np.copy(centroids)
        iteration += 1

        updateLabels(dataSet,centroids)

        centroids = getCentroids(dataSet,k)

    return dataSet,centroids

def shouldStop(oldCentroids,centroids,iteration,maxIt):
    if iteration>maxIt:
        return True
    return np.array_equal(oldCentroids,centroids)

def updateLabels(dataSet,centroids):
    numPoints, numDim = dataSet.shape
    for i in range(0,numPoints):
        dataSet[i,-1] = getLabelFromClosesetCentroid(dataSet[i,:-1],centroids)

def getLabelFromClosesetCentroid(dataSetRow,centroids):
    label = centroids[0,-1]
    minDist = np.linalg.norm(dataSetRow-centroids[0,:-1])
    for i in range(1,centroids.shape[0]):
        dist = np.linalg.norm(dataSetRow-centroids[i,:-1])
        if dist<minDist:
            minDist = dist
            label = centroids[i,-1]
    return label

def getCentroids(dataSet,k):
    result = np.zeros([k,dataSet.shape[1]])
    for i in range(1,k+1):
        oneCluster = dataSet[dataSet[:,-1]==i,:-1]
        result[i-1,:-1] = np.mean(oneCluster,axis=0)
        result[i-1,-1] = i
    return result

x1 = np.array([1,1])
x2 = np.array([1,2])
x3 = np.array([4,3])
x4 = np.array([5,4])
testx = np.vstack((x1,x2,x3,x4))

result,centroids = kmeans(testx,2,10)
# print(result)
# print(centroids)
km = KMeans(n_clusters=2,random_state=0)
km.fit(testx)
print(km.labels_)
print(km.cluster_centers_)

