import csv
import numpy as np

def to_dp(l):
    return np.array([float(x) for x in l.split(',')])

with open('./W1D1data.txt', 'r') as f:
    lines = f.readlines()
    dps = [to_dp(l) for l in lines]
    dim = len(dps[0])

    # p1
    mean = ans = np.mean(dps, axis=0)
    np.savetxt('C1.csv', [ans], delimiter=',')

    # p2
    ans = np.var(dps, axis=0)
    np.savetxt('C2.csv', [ans], delimiter=',')

    # p3
    ans = []
    for i in range(dim):
        min_dis = float('inf')
        for dp in dps:
            dis = abs(dp[i])
            if dis < min_dis:
                min_dis = dis
        ans.append(min_dis)

    np.savetxt('C3.csv', [ans], delimiter=',')


    # p4
    ans = float('inf')
    v = np.array([1.0]*dim)
    for dp in dps:
        dis = np.linalg.norm(v-dp)
        if dis < ans:
            ans = dis
    np.savetxt('C4.csv', [ans], delimiter=',')

    # p5
    m = np.matrix(dps)
    np.savetxt('C5.csv', m*np.matrix.transpose(m), delimiter=',')

    # p6
    dps = [dp-mean for dp in dps]
    max_norm = max([np.linalg.norm(dp) for dp in dps])
    np.savetxt('C6.csv', [dp/max_norm for dp in dps], delimiter=',')


