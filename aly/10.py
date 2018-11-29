import numpy as np
import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5, 6]
Y = [2.6, 3.4, 4.7, 5.5, 6.47, 7.8]

z1 = np.polyfit(X, Y, 1)
print(z1)
p1 = np.poly1d(z1)
print(p1(20))



x = np.arange(1, 7, 1)
y = z1[0]*x+z1[1]
print(y)

plt.figure()
plt.scatter(X,Y)
plt.plot(x,y)
plt.show()