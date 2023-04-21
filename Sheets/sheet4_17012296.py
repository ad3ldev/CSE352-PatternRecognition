import numpy as np
import math

c1 = np.array([
    [2, 6],
    [3, 5],
    [4, 4],
    [5, 3],
    [6, 2],
    [6, 4],
    [6, 6],
    [8, 4],
    [9, 3],
    [9, 2],
])
c2 = np.array([
    [3, 3],
    [4, 3],
    [4, 5],
    [5, 5],
    [7, 5],
    [7, 4],
    [7, 3],
])
c3 = np.array([
    [7, 2],
    [10, 1],
    [10, 3],
    [10, 5],
    [11, 3],
    [11, 4],
    [12, 2],
    [13, 5],
])

prior_c1 = c1.shape[0]/25
prior_c2 = c2.shape[0]/25
prior_c3 = c3.shape[0]/25

mean_c1 = np.mean(c1, axis=0)
mean_c2 = np.mean(c2, axis=0)
mean_c3 = np.mean(c3, axis=0)

var_c1 = np.var(c1, axis=0)
var_c2 = np.var(c2, axis=0)
var_c3 = np.var(c3, axis=0)

z1 = c1 - mean_c1
z2 = c2 - mean_c2
z3 = c3 - mean_c3

cov_c1 = np.dot(z1.T, z1)/c1.shape[0]
cov_c2 = np.dot(z2.T, z2)/c2.shape[0]
cov_c3 = np.dot(z3.T, z3)/c3.shape[0]


def prop(x, y, mean, var):
    p_x = (1/math.sqrt(2*math.pi*var[0])) * \
        (math.e**(-((x-mean[0])**2/(2*var[0]))))
    p_y = (1/math.sqrt(2*math.pi*var[1])) * \
        (math.e**(-((y-mean[1])**2/(2*var[1]))))
    return p_x * p_y


means = [mean_c1, mean_c2, mean_c3]
vars = [var_c1, var_c2, var_c3]
p1 = [6, 5]
p2 = [9, 4]
p3 = [8, 5]
points = [p1, p2, p3]

for p in points:
    for i in range(3):
        print(f"Point = {p}, Mean = {means[i]}, Var = {vars[i]}")
        print(prop(p[0], p[1], means[i], vars[i]))
        print("#############################")
