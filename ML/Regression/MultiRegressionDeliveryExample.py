import numpy as np
import matplotlib.pyplot as plt

# seed = np.random.seed(100)

# x = 2*np.random.rand(100,3)
x = 2*np.random.rand(100,1)

y = 4+3*np.random.rand(100,1) +np.random.rand(100,1)
x_b = np.c_[np.ones((100,1)),x]
# print(x_b)
theta_best = np.linalg.inv(np.dot(x_b.T,x_b)).dot(x_b.T).dot(y)
# print(theta_best)

x_new = np.array([[1.5],[1.8]])
x_new_b = np.c_[np.ones((2,1)),x_new]
print(x_new_b)

y_predicted = np.dot(x_new_b,theta_best)

print(y_predicted)

temp = [0,2]
test = theta_best[0]+theta_best[1]*temp

plt.scatter(x_new,y_predicted)
plt.scatter(x,y)
plt.plot(temp,test,'r-')

plt.show()
