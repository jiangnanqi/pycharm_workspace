from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
import numpy as np

allElectronicsData = open(r'AllElectronics.csv','rt')
reader = csv.reader(allElectronicsData)

headers = next(reader)
# print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[-1])
    rowDict = {}
    for i in range(1,len(row)-1):
        rowDict[headers[i]]=row[i]
    featureList.append(rowDict)

# print(featureList)
# print(labelList)

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
# print(dummyX)

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
# print(dummyY)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(dummyX,dummyY)
print(clf)

with open("allElectronicInformationGainOri.dot",'w') as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file= f)
# dot -Tpdf allElectronicInformationGainOri.dot -o output.pdf

oneRowX = dummyX[0]
oneRowX = oneRowX[np.newaxis,:]

oneRowX[0][0] = 1
oneRowX[0][2] = 0

predictedY = clf.predict(oneRowX)
print(predictedY)