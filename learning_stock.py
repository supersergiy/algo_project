import numpy as np
import math
import random
import scipy.io

VOLATILITY = 0
GRADIENT = 1
PREV_CHANGE = 2

STOCK_NUMBER = 490
DEFAULT_ETA = 1
DIAMETER = math.sqrt(2)
TIME_STEPS = 1000
DO_CORRELATIONS = False
DO_FORWARD_CORRELATIONS = False
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
        self.individual_eta = None
        self.stock_correlation = np.zeros((STOCK_NUMBER, STOCK_NUMBER), dtype = np.float64)
        self.forward_correlation = np.zeros((STOCK_NUMBER, STOCK_NUMBER), dtype = np.float64)
        self.forward_correlation_sum = np.zeros((STOCK_NUMBER, STOCK_NUMBER), dtype = np.float64)
        self.stock_correlation_sum = np.zeros((STOCK_NUMBER, STOCK_NUMBER), dtype = np.float64)
        self.stock_mean = np.zeros(STOCK_NUMBER, dtype = np.float64)
        self.stock_variance = np.zeros(STOCK_NUMBER, dtype = np.float64)
        self.stock_m2 = np.zeros(STOCK_NUMBER, dtype = np.float64)
        #todo: windowed statistics
        self.volatility = None
        self.gradient = None
        self.point_count = 0
        self.prev_stock = None
        self.prev_stock_change = None
        self.algos = [self.volatility_based_decision, self.gradient_based_decision, self.prev_change_based_decision]

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

    def update_stock_statistics(self, curr_stock_change):

        if (DO_FORWARD_CORRELATIONS):
            for i in range(STOCK_NUMBER):
                for j in range(STOCK_NUMBER):
                    if self.prev_stock_change == None:
                        self.forward_correlation[i, j] = 0
                    else:
                        if (self.prev_stock_change[j] == 1.0):
                            if (curr_stock_change[i] == 1.0):
                                delta = 1.0
                            else:
                                delta = 0
                        else:
                            delta = (1.0 - curr_stock_change[i]) / (1.0 - self.prev_stock_change[j])
                        self.forward_correlation_sum[i, j] = self.forward_correlation_sum[i, j] + delta
                        self.forward_correlation[i, j] = self.forward_correlation_sum[i, j] / (self.point_count - 1)

        for i in range(STOCK_NUMBER):
            delta = 1.0 - curr_stock_change[i]
            self.stock_m2[i] = self.stock_m2[i] + delta ** 2
            if (self.point_count < 2):
               self.stock_variance[i] = 0
            else:
              self.stock_variance[i] = self.stock_m2[i] / (self.point_count - 1)
              self.volatility = np.sqrt(self.stock_variance)

        if (DO_CORRELATIONS):
            for i in range(STOCK_NUMBER):
                for j in range(STOCK_NUMBER):
                    if (self.point_count < 2):
                        self.stock_correlation[i, j] = 0
                    else:
                        if (curr_stock_change[j] == 1.0):
                            if (curr_stock_change[i] == 1.0):
                                delta = 1.0
                            else:
                                delta = 0
                        else:
                            delta = (1.0 - curr_stock_change[i]) / (1.0 - curr_stock_change[j])
                        self.stock_correlation_sum[i, j] = self.stock_correlation_sum[i, j] + delta
                        self.stock_correlation[i, j] = self.stock_correlation_sum[i, j] / (self.point_count - 1)
                    #self.stock_correlation[j, i] = self.stock_correlation[i, j]

        return

    def learn(self, curr_stock):
        self.point_count = self.point_count + 1
        if (self.prev_stock != None):
            self.gradient = self.gradient_oracle(curr_stock, self.prev_stock)
            curr_stock_change = np.divide(curr_stock, self.prev_stock)
            self.update_stock_statistics(curr_stock_change)

        if (self.prev_stock != None):
            self.prev_stock_change = curr_stock_change
        self.prev_stock = curr_stock
        return

    def make_decision(self, curr_money_distribution, algo):
        if self.point_count < 2:
            return self.default_distribution()
        return self.algos[algo](curr_money_distribution)

    def gradient_based_decision(self, curr_money_distribution):
        new_distribution = np.add(curr_money_distribution, self.global_eta * self.gradient)
        return self.projection_oracle(new_distribution)

    def volatility_based_decision(self, curr_money_distribution):
        self.individual_eta = 4900000 * normalized(self.volatility)
        new_distribution = np.add(curr_money_distribution, np.multiply(self.individual_eta, self.gradient))
        return self.projection_oracle(new_distribution)

    def prev_change_based_decision(self, curr_money_distribution):
        self.individual_eta = 4900000 * normalized(np.subtract(np.ones(STOCK_NUMBER), self.prev_stock_change))
        new_distribution = np.add(curr_money_distribution, np.multiply(self.individual_eta, self.gradient))

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
    used_algos = [0, 1, 2]
    for algo in used_algos:
        money = 1.0
        print "algo: ", algo
        for i in xrange(1, 1000):
            #print "Day ", i, ": ", money
            money_distribution = decider.make_decision(money_distribution, algo)
            curr_stock =  stock_value[i]
            prev_stock = stock_value[i - 1]

            decider.learn(curr_stock)

            stock_change_ratio = np.divide(curr_stock, prev_stock)
            money *= np.dot(money_distribution, stock_change_ratio)
            if (i == 800):
                print "800 money: ", money
            if (i % 100 == 0):
                continue
                #np.savetxt("correlation%d.csv" % i, decider.stock_correlation, fmt='%.4f', delimiter=',')
                #np.savetxt("volatility%d.csv" % i, decider.volatility, fmt='%.4f', delimiter=',')
                #np.savetxt("forward_correlation%d.csv" % i, decider.forward_correlation, fmt='%.4f', delimiter=',')
        decider.clear()
        print "1000 money: ", money


if __name__ == "__main__":
        main()
