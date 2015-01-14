import numpy as np
import math
import random
import scipy.io


UPDATE_INTERVAL = 10
CHANGE_INTERVAL = 100
START_DAY = 20 + 1
HANSEN = True
STOCK_NUMBER = 29
FRACTION_NUMBER = 33
DEFAULT_ETA = 1
DIAMETER = math.sqrt(2)
DO_CORRELATIONS = False
DO_FORWARD_CORRELATIONS = False

def normalized(v):
    return v/np.sum(v)

def partial_sum(y, j):
	result = 0
	for i in xrange(j):
		result += y[i]
	return result


def fix_hansen(A):
    trunc_A = [A[0][j] for j in range(39) if len(A[0][j]) >= 3000]

    stock_value = [[0 for j in range(len(trunc_A))] for i in range(3000)]

    for i in range(3000):
        for j in range(len(trunc_A)):
                    stock_value[i][j] = float(trunc_A[j][i])
    return stock_value

class StockAlgorithm:
    def __init__(self, eta, decide):
        self.global_eta = eta
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
        self.algos = [self.gradient_based_decision, self.prev_change_based_decision]
        self.decide = self.algos[decide]

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
                        self.forward_correlation[i, j] = 1.0
                    else:
                        if (self.prev_stock_change[j] == 1.0):
                            if (curr_stock_change[i] == 1.0):
                                delta = 1.0
                            else:
                                delta = 0
                        else:
                            delta = (1.0 - curr_stock_change[i]) / (1.0 - self.prev_stock_change[j])
                        self.forward_correlation[i, j] = self.forward_correlation[i, j] * (0.99) + delta * 0.01
                        #self.forward_correlation[i, j] = self.forward_correlation_sum[i, j] / (self.point_count - 1)
        return
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

    def make_decision(self, curr_money_distribution):
        if self.point_count < 2:
            return self.default_distribution()
        return self.decide(curr_money_distribution)

    def default_distribution(self):
        result = np.ones(STOCK_NUMBER)
        return normalized(result)
    def clear(self):
        self.__init__()
        return


    def gradient_based_decision(self, curr_money_distribution):
        #print "grad"
        new_distribution = np.add(curr_money_distribution, self.global_eta * self.gradient)
        return self.projection_oracle(new_distribution)

    def volatility_based_decision(self, curr_money_distribution):
        self.individual_eta = 4900000 * normalized(self.volatility)
        new_distribution = np.add(curr_money_distribution, np.multiply(self.individual_eta, self.gradient))
        return self.projection_oracle(new_distribution)

    def prev_change_based_decision(self, curr_money_distribution):
        if (self.prev_stock_change == None):
            return self.default_distribution()
        self.individual_eta = 490 * normalized(np.subtract(np.ones(STOCK_NUMBER), self.prev_stock_change))
        new_distribution = np.add(curr_money_distribution, np.multiply(self.individual_eta, self.gradient))
        return self.projection_oracle(new_distribution)

    def forward_correlation_based_decision(self, curr_money_distribution):
        prediction_vector = np.zeros(STOCK_NUMBER)
        for i in xrange(STOCK_NUMBER):
            for j in xrange(STOCK_NUMBER):
                prediction_vector[i] = prediction_vector[i] + (self.prev_stock_change[j] - 1.0) * self.forward_correlation[i, j]
            prediction_vector[i] = prediction_vector[i] / STOCK_NUMBER
        return self.projection_oracle(np.add(curr_money_distribution, np.multiply(self.global_eta, prediction_vector)))


def main():
    data = scipy.io.loadmat('data_39_3000.mat')
    A = data['A']
    if (HANSEN):
       stock_value = fix_hansen(A)
    else:
        stock_value = zip(*A)
    result = 0
    for fraction in xrange(FRACTION_NUMBER):
        money = 1.0
        money_distribution = None
        algorithms = []
        #algorithms.append(StockAlgorithm(None, 1))
        algorithms += [StockAlgorithm(i, 0) for i in [0.001,  1, 0,  10000]]
        weights = [1.0 for i in range(len(algorithms))]
        algo_money = [1.0 for i in range(len(algorithms))]

        money_distribution = [a.default_distribution() for a in algorithms]

        curr_algo = 3
        print "Starting simulation..."
        money = 1.0

        for i in xrange(1, len(stock_value)):
            if (i % 100000 == 0):
                print "Day ", i, ": ", money

            #curr_stock =  np.array([float(x)  for x in stock_value[i]])
            #prev_stock = np.array([float(x) for x in stock_value[i - 1]])
            curr_stock = stock_value[i]
            prev_stock = stock_value[i - 1]
            stock_change_ratio = np.divide(curr_stock, prev_stock)

            for a in range(len(algorithms)):
                money_distribution[a] = algorithms[a].make_decision(money_distribution[a])[:]
                #print len(money_distribution[a]), len(stock_change_ratio)
                algo_money[a] *= np.dot(money_distribution[a], stock_change_ratio)

            if (i > START_DAY):
                money *= np.dot(money_distribution[curr_algo], stock_change_ratio)


            for a in range(len(algorithms)):
                algorithms[a].learn(curr_stock)

            flag = False
            if (i % UPDATE_INTERVAL == 0):
                best_money = max(algo_money)
                #best_algo = np.argmax(algo_money)
                for j in xrange(len(algorithms)):
                    if (algo_money[j] < 0.9 * best_money):
                        weights[j] /= 2
                algo_money = [1 for a in algorithms]

            if (i == START_DAY or i % CHANGE_INTERVAL == 0 or flag):
                #print "change"
                weights = normalized(weights)
                #print weights[0], weights[1]
                curr_algo = np.argmax(weights)
                decision = random.uniform(0, 1.0)

                #for i in xrange(len(algorithms)):
                #    money_distribution[i] = money_distribution[curr_algo]

                '''
                accumulator = 0
                curr_algo = -1
                while (accumulator < decision):
                    curr_algo += 1
                    accumulator += weights[curr_algo]
                '''
                weights = [1.0 for i in algorithms]

        result += money
        print "Fraction ", fraction, " done, money: ", money
    result = result / FRACTION_NUMBER
    print "Final Result: ", result

if __name__ == "__main__":
    main()
