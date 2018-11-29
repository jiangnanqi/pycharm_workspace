from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()
# print(iris['data'])
# print(iris['target'])

knn.fit(iris['data'],iris['target'])

predictedLable = knn.predict([[0.1,0.2,0.3,0.4]])
print(predictedLable)