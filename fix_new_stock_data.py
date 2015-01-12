data = scipy.io.loadmat('data_474_3773.mat')
A = data['A']

trunc_A = [A[0][j] for j in range(474) if len(A[0][j]) == 3773]

stock_value = [[0 for j in range(len(trunc_A))] for i in range(3773)]
for i in range(3773):
    for j in range(len(trunc_A)):
        stock_value[i][j] = float(trunc_A[j][i])