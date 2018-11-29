import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt

# data1 = [6, 7.5, 8, 0, 1]
# arr1 = np.array(data1)
# print(arr1)
# print(arr1.dtype)
#
# data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
# arr2 = np.array(data2)
# print(arr2)
# print(arr2.dtype)
#
# print(np.zeros(10))
# print(np.ones((3,6)))
# print(np.empty((2,3,2)))
#
# print(np.arange(15))
#
#
# print(np.eye(10))
# print(np.identity(5))
#
# arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
# print(arr)
# print(arr.astype(np.int32))
#
# numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
# print(numeric_strings)
# print(numeric_strings.astype(np.float32))
#
# int_array = np.arange(10)
# calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
# print(int_array,calibers)
# print(int_array.astype(calibers.dtype))
#
#
# empty_uint32 = np.empty(8, dtype=np.uint32)
# print(empty_uint32)
# print(empty_uint32.dtype)
#
# a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]])
#
# list111 = [[1.0],[2.0],[3.0],[4.0]]
# b = np.array(list111)
#
# print(a)
# print(b)
# print(a+b)
#
# arr = np.arange(10)
# print(arr)
# print(arr[5])
# print(arr[5:8])
# arr[5:8] = 10
# print(arr)
#
# names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# data = np.random.randn(7,4)
# print(names,data)
#
# print(data[names == 'Bob'])
#
#
#
# arr = np.empty((8,4))
# for i in range(8):
#     arr[i] = i
#
# print(arr)
#
# print(arr[[4,3,7,6]])
#
# arr3 = np.arange(6).reshape((2,3))
# arr4 = arr3.T
# print(arr3)
# print(arr4)
# print(np.dot(arr3,arr4))
# print(arr3.dot(arr4))
#
#
# x = np.random.randn(8)
# y = np.random.randn(8)
# print(x)
# print(y)
# print(np.minimum(x,y))

# point = np.arange(5)
# xs,ys = np.meshgrid(point,point)
# print(xs)
# print(ys)
#
# z = np.sqrt(xs**2+ys**2)
# print(z)
#
# import matplotlib.pyplot as plt
#
# plt.imshow(z)
# plt.colorbar()
# plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
# plt.show()


# xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
# yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
# cond = np.array([True, False, True, True, False])
#
# result = [(x if c else y)for x,c,y in zip(xarr,cond,yarr)]
# print(result)
#
# result1 = np.where(cond,xarr,yarr)
# print(result1.dtype)
#
#
# arr = np.random.randn(4,4)
# print(arr)
# arr = np.where(arr>0,2,-2)
# print(arr)

# arr = np.random.randn(5,4)
# print(arr)
# print(arr.mean())
# print(arr.sum())
#
# print(arr.sum(axis=1))
# print(arr.sum(axis=0))

# a = np.array([[3,7],[9,1]])
# print(a)
# print(np.sort(a))
# print(np.sort(a,axis=0))
# print(np.sort(a,axis=1))
#
# dt = np.dtype([('name','S10'),('age',int)])
# a = np.array([("raju",21),("anil",25),("ravi",  17),  ("amar",27)],dtype=dt)
# print(a)
# print(np.sort(a,order='name'))


# arr = np.random.randn(5,3)
# print(arr)
# print(np.argsort(arr))
# print(np.argsort(arr,axis=0))

# x = np.array([3,1,2])
# y = x.argsort()
# print(y)
# print(x[y])


# names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# print(np.unique(names))
# ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
# print(np.unique(ints))

# arr = np.arange(10)
# np.save('some_array', arr)

# print(np.load('some_array.npy'))


# x = np.array([[1., 2., 3.], [4., 5., 6.]])
# print(x @ np.ones(3))

# x = np.random.randn(5,5)
# xt = x.T
# print(xt)
# mat = np.dot(x,xt)
# print(mat)
# print(np.linalg.inv(mat))
# print(mat.dot(np.linalg.inv(mat)))

# nsteps = 1000
# draws = np.random.randint(0,2,nsteps)
# steps = np.where(draws>0,1,-1)
# walk = steps.cumsum()
# plt.plot(walk[:100])
# plt.show()
#
# print(walk.min())
# print(walk.max())
# print((np.abs(walk)>=10).argmax())



