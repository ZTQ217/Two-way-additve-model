import numpy as np
import pandas as pd
import math
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import reduce
from model_setnotwo_bino import Datagenerate
from scipy.optimize import minimize, fsolve
from scipy.optimize import SR1, LinearConstraint


result_list = []


def log_result(a):
    result_list.append(a)


def cons(para):
    C_new = para[range(1, N_s+1)]
    P_new = para[(N_s+1):len(para)]
    conspk = np.sum(P_new) - 1
    consck = np.dot(C_new, P_new)
    cons = np.append(consck, conspk)
    return cons


def gtheta(para):
    mu_new = para[0]
    C_new = para[range(1, N_s+1)]
    P_new = para[(N_s+1):len(para)]
    sumk = 0
    for k in np.arange(N_s):
        Ck = C_new[k]
        Pk = P_new[k]
        ak = Aij[k]
        theta_all = mu_new + X * Ck
        theta_new = theta_all[R != 0]
        logyi = ak * (Ynozero * theta_new - Binozero * np.log(1 + np.exp(theta_new)))
        Logyi = np.sum(logyi)
        logpk = np.sum(ak) * np.log(Pk)
        loglikeli = Logyi + logpk
        sumk = sumk + loglikeli
    return -sumk


def likelihood(mu_t, C_t, P_t):
    logone = 0
    for k in np.arange(N_s):
        ck = C_t[k]
        pk = P_t[k]
        theta_all = mu_t + X * ck
        theta_ob = theta_all[R != 0]
        like = np.exp(theta_ob) ** Ynozero * (1/(1 + np.exp(theta_ob))) ** Binozero
        logone = logone + like * pk
    result = np.sum(np.log(logone))
    return result


def initializer():
    global n, m, An, Am, Anm, bmu, N_s, prob, C_inter, Label, r_true
    n = 50
    m = 40
    N_s = 2
    r = 5
    Label = np.arange(N_s) + 1
    # prob = np.random.uniform(0, 1, N_s)
    # prob = prob/np.sum(prob)
    bmu = 3.5
    prob = np.array([0.4, 0.6])
    An = np.full(n, 1)
    np.random.seed([1024])
    Am = np.full(m, 1)
    Anm = pd.DataFrame((np.zeros(n * m) + 1).reshape((n, m)), dtype=float)
    Cinter_l = np.random.uniform(-2, 2, N_s - 1) * r
    C_last = -np.dot(prob[np.arange(N_s - 1)], Cinter_l) / prob[N_s - 1]
    C_inter = np.append(Cinter_l, C_last)
    C_inter = np.array(C_inter)
    r_true = [bmu]
    r_true.extend(prob[0:(N_s - 1)])
    r_true.extend(C_inter[0:(N_s - 1)])


def algorithm(rep):
    print("rep", rep)
    global sample_number, Ynozero, Binozero, X, R, Aij, Cij
    Data = Datagenerate(n, m, bmu, prob, C_inter, Label, rep)
    X = Data["X"]
    Y = Data["Y"]
    Bi = Data["Bi"]
    R = Data["R"]
    #Bi = np.zeros(n*m).reshape(n, m) + 5
    sample_number = len(np.where(R != 0)[0])
    print(sample_number)
    set_of_Y = np.delete(np.unique(Y), np.where(np.unique(Y) == 0))
    Ynozero = Y[R != 0]
    Binozero = Bi[R != 0]
    print(Ynozero[0], Binozero[0])
    # model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    # b_init = model.fit(X[R ==1], Y[R == 1]).coef_
    C_0 = C_inter
    P_0 = prob
    mu_0 = bmu
    #####under bernuolli situation#####
    diff = 1000
    ltheta_0 = 10000
    C_t = C_0
    P_t = P_0
    mu_t = mu_0
    ltheta_t = ltheta_0
    Ltheta = []


    while diff > 0.01:
        aij = []
        sumak = np.zeros(sample_number)
        for k in np.arange(N_s):
            C = C_t[k]
            p = P_t[k]
            theta_t = mu_t + X * C
            thetat_ob = theta_t[R != 0]
            Py1 = np.exp(thetat_ob) / (1 + np.exp(thetat_ob))
            Py0 = 1 - Py1
            Pyp = Py1 ** Ynozero * (Py0 ** (Binozero-Ynozero)) * p
            aij.append(Pyp)
            sumak = sumak + Pyp
        Aij = [a/sumak for a in aij]
        """
        Pnew = []
        for k in np.arange(N_s):
            ak = [a[k] for a in Aij]
            pk = reduce(lambda a, b: a+b, ak)
            Pnew.append(pk/sample_number)
        P_t = Pnew
        C_s = C_t
        diff_2 = 10
        diff_1 = 10
        while diff_2 > 0.001:
            dernew = []
            Csnew = []
            for k in np.arange(N_s):
                c_k = C_s[k]
                ak = np.array(Aij)[:, k]
                phik = np.exp(Xvec * c_k)/(1 + np.exp(Xvec * c_k))
                dbetak = ak * (Yvec * Xvec - Bivec * phik * Xvec)
                derbeta = np.mean(dbetak)
                print("k = ", k)
                print("derbeta", derbeta)
                secdbeta = ak * Bivec * phik * (1 - phik) * (Xvec ** 2)
                secderbeta = np.mean(secdbeta)
                print("secderbeta", secderbeta)
                ck_updata = c_k + derbeta / secderbeta
                Csnew.append(ck_updata)
                dernew.append(derbeta)
            diff_2 = np.max(np.abs(dernew))
            diff_1 = np.max(np.abs(np.array(Csnew)-C_s))
            C_s = np.array(Csnew)
            print("C_new", C_s)
        C_t = C_s
        ltheta_new = likelihood(C_0, P_0)
        print("ltheta", ltheta_new)
        diff = np.abs(ltheta_new - ltheta_t)
        ltheta_t = ltheta_new
        """
        para_0 = mu_t
        para_0 = np.append(para_0, C_t)
        para_0 = np.append(para_0, P_t)
#        print(cons(para_0))
        eq_cons = {'type': 'eq',
                   'fun': cons}
        Para_t = minimize(gtheta, para_0, constraints=eq_cons)
        para_t = Para_t.x
        mu_t = para_t[0]
#        print("mu_t =", mu_t)
        C_t = para_t[range(1, N_s+1)]
#        print("C_t = ", C_t)
        P_t = para_t[(N_s+1):len(para_t)]
        P_t = np.sort(P_t)
#        print("P_t:", P_t)
        ltheta_new = likelihood(mu_t, C_t, P_t)
#        print("ltheta", ltheta_new)
        diff = np.abs(ltheta_new - ltheta_t)
        ltheta_t = ltheta_new
        Ltheta.append(ltheta_t)
    hat_mu = mu_t
    print("mu_0", bmu)
    print("mu_t", hat_mu)
    hat_P = P_t
    print("C_inter", C_inter)
    print("C_t", C_t)
    hat_C = C_t
    print("prob", prob)
    print("P_t", P_t)
    result = dict(zip(["hat_mu", "hat_P", "hat_C", "sample_number"],
                      [hat_mu, hat_P, hat_C, sample_number]))
    return result


if __name__ == '__main__':
    core = multiprocessing.cpu_count()
    initializer()
    p = Pool(core, initializer, ())
    N_rep = 100
    for rep in range(N_rep):
        Sim_results = p.apply_async(algorithm, args=(rep,), callback=log_result)
        Sim_results.get()
    p.close()
    p.join()
    mu_hat = []
    mu_bias = []
    C_hat = []
    C_bias = []
    Prob_hat = []
    Prob_bias = []
    N_sample = []
    for a in result_list:
        mu_hat.append(a["hat_mu"])
        mu_bias.append(np.abs(a["hat_mu"]-bmu))
        C_hat.append(a["hat_C"])
        C_bias.append(np.abs(a["hat_C"]-C_inter))
        Prob_hat.append(a["hat_P"])
        Prob_bias.append(np.abs(a["hat_P"]-prob))
        N_sample.append(a["sample_number"])
    n_sample = np.mean(N_sample)
    names = ['theta_true', 'mean(hat_theta)', 'median(hat_theta)', 'sd(hat_btheta)',
                                     'mean of bias', 'rmse']
    muhat_mean = np.mean(mu_hat)
    muhat_median = np.median(mu_hat)
    muhat_sd = np.std(mu_hat)
    mu_biasmean = np.mean(np.array(mu_bias))
    mu_rmse = np.sqrt(np.mean(np.array(mu_bias)**2))
    mu_est = [bmu, muhat_mean, muhat_median, muhat_sd, mu_biasmean, mu_rmse]
    data_mu = dict(zip(names, mu_est))
    Chat_mean = np.mean(np.array(C_hat), axis=0)
    Chat_median = np.median(np.array(C_hat), axis=0)
    Chat_sd = np.std(np.array(C_hat), axis=0)
    C_biasmean = np.mean(np.array(C_bias), axis=0)
    C_rmse = np.sqrt(np.mean(np.array(C_bias) ** 2, axis=0))
    C_est = [C_inter, Chat_mean, Chat_median, Chat_sd, C_biasmean, C_rmse]
    data_C = [dict(zip(names, list(a))) for a in np.array(C_est).T]
    Phat_mean = np.mean(np.array(Prob_hat), axis=0)
    Phat_median = np.median(np.array(Prob_hat), axis=0)
    Phat_sd = np.std(np.array(Prob_hat), axis=0)
    P_biasmean = np.mean(np.array(Prob_bias), axis=0)
    P_rmse = np.sqrt(np.mean(np.array(Prob_bias) ** 2, axis=0))
    P_est = [prob, Phat_mean, Phat_median, Phat_sd, P_biasmean, P_rmse]
    data_P = [dict(zip(names, list(a))) for a in np.array(P_est).T]
    r_est = pd.DataFrame(columns=['theta_true', 'mean(hat_theta)', 'median(hat_theta)', 'sd(hat_btheta)',
                                     'mean of bias', 'rmse'])
    r_est = r_est.append(data_mu, ignore_index=True)
    data_C.extend(data_P)
    for g in data_C:
        r_est = r_est.append(g, ignore_index=True)
    r_est.to_csv("./groupGLM_missing={}%_fulllike_data_n={}_m={}_settings.csv".format(int(n_sample/(n*m)*100), n, m), index=True, header=True)
