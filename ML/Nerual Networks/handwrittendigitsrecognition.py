import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from nerualnetwork import NerualNetwork
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

digits = load_digits()
# print(digits)
x = digits.data
y = digits.target
#减去最小，除以最大，将数据归一化为0~1之间  神经网络一般这样输入
x -= x.min()
x /= x.max()
# print(x)
nn = NerualNetwork([64,40,10],'logistic')
x_train,x_test,y_train,y_test = train_test_split(x,y)
label_train = LabelBinarizer().fit_transform(y_train)
label_test = LabelBinarizer().fit_transform(y_test)

nn.fit(x_train,label_train)
predictions = []

for i in range(x_test.shape[0]):
    o = nn.predict(x_test[i])
    predictions.append(np.argmax(o))
# print(predictions)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))