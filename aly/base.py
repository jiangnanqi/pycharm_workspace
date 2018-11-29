import numpy as np
#
# layers = [4,3,2]
# weights = []
# for i in range(1, len(layers)):
#
#     weights.append(np.random.random((layers[i - 1] + 1, layers[i] + 1)))
#
# print(weights)

p = [[-1,1,1],[1,0,2],[1,1,-1]]
p = np.array(p)
print(np.linalg.det(p))
print(np.linalg.inv(p))
print(np.linalg.det(p)*np.linalg.inv(p))
print('============')
aa = [[1,0,0],[0,2,0],[0,0,-3]]
a = p*aa*np.linalg.inv(p)
# print(a**3+2*aa**2-3*a)
print(np.linalg.matrix_rank(p))
print(np.linalg.eig(p))
print('=============')
x = np.diag([1,2,3])
print(np.linalg.eig(x)[1])

xx = [[-1,1,0],[-4,3,0],[1,0,2]]
print(np.linalg.eig(xx))