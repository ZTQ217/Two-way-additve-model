import pandas as pd
import numpy as np
import math
from functools import reduce
from scipy.stats import norm


def drnf(x, p, n):
    z = []
    ps = p.cumsum(0)
    r = np.random.uniform(0, 1, n)
    for i in np.arange(n):
        z.append(x[np.where(r[i] <= ps)[0][0]])
    return z


def P_last(P, n, m):
        y = np.zeros(n * m).reshape(n, m)
        for j in np.arange(m):
            for i in np.arange(n):
                p_last = list(map(lambda a: a[j][i], P))
                y[i][j] = np.argmax(np.array(p_last)) + 1
        return y


def Datagenerate(n, m, An, Am, bmu, prob, C_inter, Label, alpha, beta, rep):
    Bi = np.zeros(n*m).reshape(n, m) + 5
    np.random.seed([50 + n + m + rep])
    X = np.random.uniform(-0.5, 1, n*m).reshape(n, m)
    f = lambda x: C_inter[x - 1]
    M_Label = np.array(drnf(Label, prob, (n * m)))
    G_Gamma = pd.Series(M_Label).apply(f)
    Gamma = np.array(G_Gamma).reshape((n, m))
    thetamu = bmu + np.dot(alpha.reshape(n, 1), Am.reshape(1, m)) + np.dot(An.reshape(n, 1), beta.reshape(1, m)) + Gamma * X
    Prob = np.exp(thetamu)/(1+np.exp(thetamu))
    p_generate = np.vectorize(lambda b, a: np.random.binomial(b, a, 1))
    Y = p_generate(Bi, Prob)
    Y_ad = Y ** 2 - 2 * Y - 0.4
    p_in_B = pd.DataFrame(Y_ad).applymap(lambda a: norm.cdf(a, loc=0, scale=1))
    pR_generate = np.vectorize(lambda a: np.random.binomial(1, a, 1))

    R = pR_generate(p_in_B.values)
    result = dict(zip(["Y", "Y*R", "Bi", "R", "X", "bmu", "C_inter", "prob", "alpha", "beta"], [Y, Y * R, Bi, R, X, bmu, C_inter, prob, alpha, beta]))
    return result



