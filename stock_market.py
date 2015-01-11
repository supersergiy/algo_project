#! /usr/bin/env python
import numpy
import scipy.io
import math


# L1 normalize the vector such that the sum of all components is 1
def normalize(vector):
    total = sum(vector)
    for i in range(len(vector)):
        vector[i] /= total
    return vector


# dot product of two vectors, assuming dimensions are equal
def dotProduct(vec1, vec2):
    return sum(vec1[i]*vec2[i] for i in range(len(vec1)))

def simpleProjection(y):
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
    
#grabs the gradient of the stocks at day 'day' and point 'portfolio'
def gradientOracle(dayRecords, day, portfolio):
    stocks = len(portfolio)
    
    ratio = [dayRecords[day][stock] / dayRecords[day-1][stock] for stock in range(stocks)]
    rDotX = dotProduct(ratio, portfolio)
    
    gradient = [ratio[i] / 1 for i in range(stocks)]
    return gradient


def updateWealth(wealth, dayRecords, day, portfolio):
    stocks = len(dayRecords[day])
    
    ratio = [dayRecords[day][stock] / dayRecords[day-1][stock] for stock in range(stocks)]
    wealthPerStock = [portfolio[i]*wealth for i in range(stocks)]
    return dotProduct(wealthPerStock, ratio)

def findOptEta(initialWealth, dayRecords, portfolio, day):
    stocks = len(portfolio)
    # run gradient descent!
    best_wealth = -1
    best_portfolio = []
    best_eta = -1
    print str(initialWealth) + ' initialWealth'
    for eta in range(10, 400, 10):
        wealth = initialWealth
        for t in range(day, min(day + timestep, 1000), 1):
            #print 'day ' + str(t)
            wealth = updateWealth(wealth, dayRecords, t, portfolio)
            gradient = gradientOracle(dayRecords, t, portfolio)
            portfolio = [ portfolio[i] + eta*gradient[i] for i in range(stocks) ]
            portfolio = simpleProjection(portfolio)
        if (wealth > best_wealth):
            best_wealth = wealth
            best_portfolio = portfolio
            best_eta = eta
        print wealth,eta
    print "best wealth: %0.5f\n" % best_wealth, "best eta: ", best_eta
    return [best_wealth, best_portfolio, best_eta, best_wealth/initialWealth] 

def start():
    matfile = scipy.io.loadmat('data_490_1000.mat')
    data = matfile['A']
    # data is a list of arrays. each array corresponds to one stock.
    stocks = len(data)
    print stocks
    days = len(data[0])
    counter = [0]*stocks
    updown = [[0]*days]*stocks

    # transpose the data such that each array corresponds to one day
    dayRecords = [data[:,i] for i in range(days)]

    # begin with a uniform portfolio
    portfolio = normalize([1.0]*stocks)
    print portfolio

    #arbitrary starting wealth
    wealth = 1.0

    #G = float(raw_input("enter G:"))
    # diameter D of the set of probability vectors is sqrt(2). a reasonable G is 10
    
    eta = math.sqrt(2.0)/(1*math.sqrt(days*1.0))
    
    global timestep 
    timestep = 20
    print "timestep %d" %(timestep)
    month_num = days / timestep
    print str(month_num) + " months"
    opt_eta = [-1] * month_num
    opt_ratio = [-1] * month_num
    with open("gradient.csv", "w") as of:
        
        month = 0
        for month in range(month_num):
            print "month " + str(month)
            day = month*timestep+1
            results = findOptEta(wealth, dayRecords, portfolio, day)
            wealth = results[0]
            print wealth
            #portfolio = results[1]
            print portfolio
            opt_eta[month] = results[2]
            print opt_eta
            opt_ratio[month] = results[3]
            print opt_ratio
            

        exit()


            

        exit()

        # run gradient descent!
        best_wealth = -1
        best_eta = -1

        for eta in range(0, 400, 10):
            wealth = 1
            for t in range(1, 31-1):
                wealth = updateWealth(wealth, dayRecords, t, portfolio)
                
                gradient = gradientOracle(dayRecords, t, portfolio)
                for i in range(stocks):
                    actualR = dayRecords[t+1][i] / dayRecords[t][i]
                    if actualR > 1:
                        updown[i][t] = 1
                    if actualR < 1:
                        updown[i][t] = -1
                    if gradient[i] > 1 and actualR < 1:
                        #counter[i] += 1
                        counter[i] += abs(gradient[i] - actualR)
                    if gradient[i] < 1 and actualR > 1:
                        #counter[i] += 1
                        counter[i] += abs(gradient[i] - actualR)
                #of.write(str(gradient) + '\n')
                if t%100 == 0:
                    print t
                    #of.write(str(counter) + '\n')
                    mispredictDiff = sum(counter[i] for i in range(stocks))
                    print mispredictDiff
                    total = sum(dayRecords[t][i] for i in range(stocks))
                    print total
                    counter = [0]*stocks
                    print wealth
                    #wealth = 1.0
                portfolio = [ portfolio[i] + eta*gradient[i] for i in range(stocks) ]
                portfolio = simpleProjection(portfolio)
            for stock in range(stocks):
                of.write(str(updown[stock]) + '\n')
            
            if (wealth > best_wealth):
                best_wealth = wealth
                best_eta = eta
            
        print "best wealth: %0.5f\n" % best_wealth, "best eta: ", best_eta


    print "Eta = %f" % eta
    print "Starting fraction of wealth on each stock: %0.5f" % (1.0/stocks)
    print "--After running for %d rounds--" % days
    print "Max fraction of wealth on any stock: %0.5f" % max(portfolio)
    print "Min fraction of wealth on any stock: %0.5f" % min(portfolio)
    print "Ratio of final wealth to starting wealth: %0.5f" % wealth
        

start()

