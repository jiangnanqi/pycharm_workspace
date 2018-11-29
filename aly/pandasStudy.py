import numpy as np
import pandas as pd

# arr1 = np.arange(10)
# print(arr1)
#
# s1 = pd.Series(arr1)
# print(s1)
#
# dic1 = {'a':10,'b':20,'c':30,'d':40,'e':50}
# print(dic1)
#
# s2 = pd.Series(dic1)
# print(s2)
#
# arr2 = np.arange(12).reshape(4,3)
# print(arr2)
# s3 = pd.DataFrame(arr2)
# print(s3)
# print(s3[0][0])
#
# dic2 = {'a':[1,2,3,4],'b':[5,6,7,8],'c':[9,10,11,12],'d':[13,14,15,16]}
# s4 = pd.DataFrame(dic2)
# print(s4)
#
# s4 = pd.Series(np.array([1,1,2,3,5,8]))
# print(s4)
# print(s4.index)
#
# print(s4[[1,3,5]])

stu_dic = {'Age':[14,13,13,14,14,12,12,15,13,12,11,14,12,15,16,12,15,11,15],
'Height':[69,56.5,65.3,62.8,63.5,57.3,59.8,62.5,62.5,59,51.3,64.3,56.3,66.5,72,64.8,67,57.5,66.5],
'Name':['Alfred','Alice','Barbara','Carol','Henry','James','Jane','Janet','Jeffrey','John','Joyce','Judy','Louise','Marry','Philip','Robert','Ronald','Thomas','Willam'],
'Sex':['M','F','F','F','M','M','F','F','M','M','F','F','F','F','M','M','M','M','M'],
'Weight':[112.5,84,98,102.5,102.5,83,84.5,112.5,84,99.5,50.5,90,77,112,150,128,133,85,112]}
student = pd.DataFrame(stu_dic)
#
# print(student.head())
# print(student.tail())
# print(student.loc[[0,2,4,5,7]])
# print(student[['Name','Height','Weight']])
# print(student.loc[:,['Name','Height','Weight']].head())
# print(student[(student['Age']>12)&(student['Sex']=='F')][['Name','Height','Weight']])

np.random.seed(1234)
d1 = pd.Series(2*np.random.normal(size = 100)+3)
d2 = np.random.f(2,4,size = 100)
d3 = np.random.randint(1,100,size = 100)



def stats(x):
	return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),x.quantile(.75),
                      x.mean(),x.max(),x.idxmax(),x.mad(),x.var(),x.std(),x.skew(),x.kurt()],
                     index = ['Count','Min','Whicn_Min','Q1','Median','Q3','Mean','Max',
                              'Which_Max','Mad','Var','Std','Skew','Kurt'])
print(stats(d1))

print(student['Sex'].describe())








