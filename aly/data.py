from sklearn import preprocessing
import numpy as np

ccc = [1,3,5,2,4,3,2,5,1,3,5,4,2,5]

# lb = preprocessing.MultiLabelBinarizer(classes=['a','b','c','d'],sparse_output=True)
# a = lb.fit_transform(ccc).toarray()
# print(a)


lb2 = preprocessing.OneHotEncoder()
b = lb2.fit_transform(ccc).toarray()
print(b)