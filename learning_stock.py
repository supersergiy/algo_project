import numpy as np
import random
import scipy.io
from sklearn.preprocessing import normalize

STOCK_NUMBER = 490
DEFAULT_ETA = 1
DO_CORRELATIONS = True
def normalized(v):
    return v/np.sum(v)

def partial_sum(y, j):
	result = 0
	for i in xrange(j):
		result += y[i]
	return result



class StockAlgorithm:
    def __init__(self):
        self.global_eta = DEFAULT_ETA
        self.stock_correlation = np.zeros((STOCK_NUMBER, STOCK_NUMBER), dtype = np.float64)#todo
        self.stock_volatility = np.zeros(STOCK_NUMBER, dtype = np.float64)#todo
        self.stock_mean = np.zeros(STOCK_NUMBER, dtype = np.float64)#done
        self.stock_variance = np.zeros(STOCK_NUMBER, dtype = np.float64)#done
        self.stock_m2 = np.zeros(STOCK_NUMBER, dtype = np.float64)#done
        #todo: windowed statistics
        #todo: statistics for normalized stock values
        self.gradient = None
        self.point_count = 0
        self.prev_stock = None

    def gradient_oracle(self, curr_stock, prev_stock):
        return np.divide(curr_stock, prev_stock)

    def projection_oracle(self, vector):
        u = sorted(vector, reverse = True)
        rou = float("-inf");
        acc = 0;

        for j in xrange(len(u)):
            acc += u[j]
            if (u[j] + (1.0 - acc) / (j + 1.0)) > 0:
                rou = j + 1.0

        acc = partial_sum(u, int(rou))

        lam = (1.0 - acc) / rou

        x = [max(vector[i] + lam, 0.0) for i in xrange(len(vector))]

        if (np.sum(x) - 1.0 > 0.00001):
            x = normalized(x)
        return x

    def update_stock_statistics(self, curr_stock):
        self.point_count = self.point_count + 1
        for i in range(STOCK_NUMBER):
            delta = curr_stock[i] - self.stock_mean[i]
            self.stock_mean[i] = self.stock_mean[i] + delta / self.point_count
            self.stock_m2[i] = self.stock_m2[i] + delta ** 2
            if (self.point_count < 2):
               self.stock_variance[i] = 0
            else:
              self.stock_variance[i] = self.stock_m2[i] / (self.point_count - 1)
        if (DO_CORRELATIONS):
            for i in range(STOCK_NUMBER):
                for j in range(i):
                    if (self.stock_variance[i] == 0 or self.stock_variance[j] == 0):
                        self.stock_correlation[i, j] = 0
                    else:
                        delta = self.stock_correlation[i, j] - (curr_stock[i] - self.stock_mean[i]) * (curr_stock[j] - self.stock_mean[j])/ ((self.stock_variance[i] * self.stock_variance[j]) ** 1/2)
                        self.stock_correlation[i, j] = self.stock_correlation[i, j] + delta / (self.point_count - 1)
                    self.stock_correlation[j, i] = self.stock_correlation[i, j]
        return

    def learn(self, curr_stock):
        if (self.prev_stock != None):
            self.gradient = self.gradient_oracle(curr_stock, self.prev_stock)
        self.update_stock_statistics(curr_stock)
        self.prev_stock = curr_stock
        return

    def make_decision(self, curr_money_distribution):
        if self.point_count < 2:
            return self.default_distribution()
        return self.gradient_based_decision(curr_money_distribution)

    def gradient_based_decision(self, curr_money_distribution):
        new_distribution = np.add(curr_money_distribution, self.global_eta * self.gradient)
        return self.projection_oracle(new_distribution)

    def default_distribution(self):
        result = np.ones(STOCK_NUMBER)
        return normalized(result)
    def clear(self):
        self.__init__()
        return

def main():
    data = scipy.io.loadmat('data_490_1000.mat')
    A = data['A']
    stock_value = zip(*A)

    money = 1.0
    money_distribution = None

    decider = StockAlgorithm()
    print "Starting simulation..."
    for i in xrange(1, 1000):
        print "Day ", i, ": ", money
        money_distribution = decider.make_decision(money_distribution)
        curr_stock =  stock_value[i]
        prev_stock = stock_value[i - 1]

        decider.learn(curr_stock)

        stock_change_ratio = np.divide(curr_stock, prev_stock)
        money *= np.dot(money_distribution, stock_change_ratio)
        if (i % 100 == 0):
            np.savetxt("stock_correlations%d.csv" % i, decider.stock_correlation, fmt='%.2f', delimiter=',')
    print money

if __name__ == "__main__":
        main()
