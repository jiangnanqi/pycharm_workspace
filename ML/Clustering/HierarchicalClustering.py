from numpy import *

class cluster_node:
    def __init__(self,vec,left=None,right=None,distance=0.0,id=None,count=1):
        self.left = left
        self.right = right
        self.vec = vec
        self.distance = distance
        self.id = id
        self.count = count

def L2dist(v1,v2):
    return sqrt(sum((v1-v2)**2))
def L1dist(v1,v2):
    return sum(abs(v1-v2))

def hcluster(features,distance=L2dist):
    distances = {}
    currentclustid  = -1

    clust = [cluster_node(array(features[i]),id=i) for i in range(0,len(features))]
    # print(clust)

    while len(clust) >1:
        lowerpair = (0,1)
        closest = distance(clust[0].vec,clust[1].vec)

        for i in range(len(clust)):
            for j in range(i+1,len(clust)):
                if(clust[i].id,clust[j].id) not in distances:
                    distances[(clust[i].id,clust[j].id)] = distance(clust[i].vec,clust[j].vec)

                d = distances[(clust[i].id,clust[j].id)]

                if d<closest:
                    closest = d
                    lowerpair = (i,j)
        mergevec = [(clust[lowerpair[0]].vec[i]+clust[lowerpair[1]].vec[i])/2.0 for i in range(len(clust[0].vec))]

        newcluster = cluster_node(array(mergevec),left=clust[lowerpair[0]],right=clust[lowerpair[1]],distance=closest,id=currentclustid)

        currentclustid -=1
        del clust[lowerpair[1]]
        del clust[lowerpair[0]]
        clust.append(newcluster)
    return clust[0]

def extract_cluster(clust,dist):
    clusters = {}

    if clust.distance <dist:
        return [clust]
    else:
        c1 = []
        c2 = []
        if clust.left != None:
            c1 = extract_cluster(clust.left,dist)
        if clust.right != None:
            c2 = extract_cluster(clust.right,dist)
        return c1+c2
def get_cluster_element(clust):
    if clust.id>=0:
        return [clust.id]
    else:
        c1 = []
        c2 = []
        if clust.left != None:
            c1 = get_cluster_element(clust.left)
        if clust.right != None:
            c2 = get_cluster_element(clust.right)
        return c1+c2


def printclust(clust, labels=None, n=0):
    # indent to make a hierarchy layout
    for i in range(n):
        print(' '),
    if clust.id < 0:
        # negative id means that this is branch
        print('-')
    else:
        # positive id means that this is an endpoint
        if labels == None:
            print(clust.id)
        else:
            print(labels[clust.id])

    # now print the right and left branches
    if clust.left != None: printclust(clust.left, labels=labels, n=n + 1)
    if clust.right != None: printclust(clust.right, labels=labels, n=n + 1)


fea = [[0,1],[2,2],[3,3],[1,1],[3,2],[3,4]]
test = hcluster(fea)
test1 = extract_cluster(test,2)
for temp in test1:
    a = get_cluster_element(temp)
    for i in a:
        print(fea[i])
    print('--------------')
