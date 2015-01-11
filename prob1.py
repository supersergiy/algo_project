import numpy as np
import random
import scipy.io
from sklearn.preprocessing import normalize



def normalized(v):
    return v/np.sum(v)

def partial_sum(y, j):
	result = 0
	for i in xrange(j):
		result += y[i]
	return result

def gradientOracle(x, s):
	grad = s
	dot = np.dot(x, s)

	return grad
def projectionOracle(y):
    u = sorted(y, reverse = True)
    pho = -1;
    s = 0;
    for j in range(len(u)):
        s = s + u[j]
        value = u[j] + (1 - s)/(j+1)
        if value > 0:
            pho = j + 1
    s = 0
    for i in range(pho):
        s = s + u[i]
    lam = (1 - s)/pho
    x = [max(y[i] + lam, 0) for i in range(len(y))]
    return x


data = scipy.io.loadmat('data_490_1000.mat')

D = 2 ** (0.5)
G = 0.0006
T = 1000

A = data['A']
A = zip(*A)

eita = D / (G * (T ** 0.5))


eita = 100.0006
print eita

x = np.ones(490)
#x = projectionOracle(x)
x = normalized(x)



money = 1.0
for i in xrange(1, 1000):
    s_i =  A[i]
    s_im1 = A[i - 1]
    s = np.divide(s_i, s_im1);

    money *= np.dot(x, s)

    y = np.add(x, eita * gradientOracle(x, s))
    x = projectionOracle(y)
print money

#print x







