import numpy as np
import random

def genData(numPoints,bias,variance):
    x = np.zeros(shape=(numPoints,2))
    y =np.zeros(shape=numPoints)

    for i in range(0,numPoints):
        x[i][0] = 1
        x[i][1] = i

        y[i] = (i+bias) + random.uniform(0,1) *variance

    return x,y

# def gradentDescent(x,y,theta,alpha,m,numIterations):
#     xTrans = x.T
#     for i in range(0,numIterations):
#         hypothesis = np.dot(x,theta)

x,y = genData(100,25,10)

