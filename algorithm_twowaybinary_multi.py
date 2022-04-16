import numpy as np
import pandas as pd
import math
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import reduce
from model_set_binary_multi import Datagenerate
from scipy.optimize import minimize, fsolve
from scipy.optimize import SR1, LinearConstraint


result_list = []


def log_result(a):
    result_list.append(a)


def cons(para):
    alpha_new = para[range(1, n+1)]
    beta_new = para[range(n+1, n+m+1)]
    C_new = para[range(n+m+1, N_s+n+m+1)]
    consalpha = np.sum(alpha_new)
    consbeta = np.sum(beta_new)
    consd = np.append(consalpha, consbeta)
    conspk = np.sum(Pnew) - 1
    consck = np.dot(C_new, Pnew)
    consr = np.append(consck, conspk)
    cons = np.append(consd, consr)
    return cons


def gtheta(para):
    mu_new = para[0]
    alpha_new = para[range(1, n+1)]
    beta_new = para[range(n+1, n+m+1)]
    C_new = para[range(n+m+1, n+m+N_s+1)]
    sumk = 0
    for k in np.arange(N_s):
        Ck = C_new[k]
        Pk = Pnew[k]
        ak = Aij[k]
        theta_new = mu_new + np.dot(alpha_new.reshape(n, 1), Am.reshape(1, m)) + np.dot(An.reshape(n, 1), beta_new.reshape(1, m)) + Xvec * Ck
        logyi = ak * (Yvec * theta_new - Bivec * np.log(1 + np.exp(theta_new)))
        Logyi = np.sum(logyi)
        logpk = np.sum(ak) * np.log(Pk)
        loglikeli = Logyi + logpk
        sumk = sumk + loglikeli
    return -sumk


def likelihood(mu_t, alpha_t, beta_t, C_t, P_t):
    logone = 0
    for k in np.arange(N_s):
        ck = C_t[k]
        pk = P_t[k]
        like = np.exp(mu_t + np.dot(alpha_t.reshape(n, 1), Am.reshape(1, m)) + np.dot(An.reshape(n, 1), beta_t.reshape(1, m)) + Xvec * ck) ** Yvec * (1/(1 + np.exp(mu_t + np.dot(alpha_t.reshape(n, 1), Am.reshape(1, m)) + np.dot(An.reshape(n, 1), beta_t.reshape(1, m)) + Xvec * ck))) ** Bivec
        logone = logone + like * pk
    result = np.sum(np.log(logone))
    return result


def initializer():
    global n, m, An, Am, Anm, bmu, N_s, prob, C_inter, Label, r_true, alpha, beta
    n = 150
    m = 100
    An = np.full(n, 1)
    Am = np.full(m, 1)
    Anm = pd.DataFrame((np.zeros(n * m) + 1).reshape((n, m)), dtype=float)
    N_s = 2
    r = 5
    Label = np.arange(N_s) + 1
    #prob = np.random.uniform(0, 1, N_s)
    #prob = prob/np.sum(prob)
    bmu = 3.5
    prob = np.array([0.4, 0.6])
    An = np.full(n, 1)
    np.random.seed([1024])
    alpha = np.random.uniform(-2, 2, n)
    alpha = alpha - np.mean(alpha)
    Am = np.full(m, 1)
    beta = np.random.uniform(-2, 2, m)
    beta = beta - np.mean(beta)
    Anm = pd.DataFrame((np.zeros(n * m) + 1).reshape((n, m)), dtype=float)
    Cinter_l = np.random.uniform(-2, 2, N_s - 1) * r
    C_last = -np.dot(prob[np.arange(N_s - 1)], Cinter_l) / prob[N_s - 1]
    C_inter = np.append(Cinter_l, C_last)
    C_inter = np.array(C_inter)
    r_true = [bmu]
    r_true.extend(prob[0:(N_s - 1)])
    r_true.extend(C_inter[0:(N_s - 1)])


def algorithm(rep):
    Data = Datagenerate(n, m, An, Am, bmu, prob, C_inter, Label, alpha, beta, rep)
    X = Data["X"]
    Y = Data["Y"]
    Bi = Data["Bi"]
    global sample_number, set_of_Y, Yvec, Xvec, Bivec
    #Bi = np.zeros(n*m).reshape(n, m) + 5
    sample_number = len(np.where(Y != 0)[0])
    print(sample_number)
    set_of_Y = np.delete(np.unique(Y), np.where(np.unique(Y) == 0))
    Yvec = Y#.reshape(n * m)
    Xvec = X#.reshape(n * m)
    Bivec = Bi#.reshape(n * m)
    #Ynozero = Yvec[np.where(Yvec != 0)]
    #Xnozero = Xvec[np.where(Yvec != 0)]
    #Binozero = Bivec[np.where(Yvec != 0)]
    C_0 = Data["C_inter"]
    P_0 = Data["prob"]
    mu_0 = Data["bmu"]
    alpha_0 = Data["alpha"]
    beta_0 = Data["beta"]
    #####under bernuolli situation#####
    diff = 1000
    ltheta_0 = 10000
    C_t = C_0
    P_t = P_0
    mu_t = mu_0
    alpha_t = alpha_0
    beta_t = beta_0
    ltheta_t = ltheta_0
    Ltheta = []
    Gtheta = []
    while diff > 0.05:
        aij = []
        sumak = np.zeros(n*m).reshape(n, m)
        for k in np.arange(N_s):
            C = C_t[k]
            p = P_t[k]
            theta_t = mu_t + np.dot(alpha_t.reshape(n, 1), Am.reshape(1, m)) + np.dot(An.reshape(n, 1), beta_t.reshape(1, m)) + Xvec * C
            Py1 = np.exp(theta_t) / (1 + np.exp(theta_t))
            Py0 = 1 - Py1
            Pyp = Py1 ** Yvec * (Py0 ** (Bivec-Yvec)) * p
            aij.append(Pyp)
            sumak = sumak + Pyp
        global Aij, Pnew
        Aij = [a/sumak for a in aij]
        Pnew = []
        for k in np.arange(N_s):
            ak = np.sum(Aij[k])
            Pnew.append(ak/sample_number)
        P_t = Pnew
        """
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
        para_0 = np.append(para_0, alpha_t)
        para_0 = np.append(para_0, beta_t)
        para_0 = np.append(para_0, C_t)
        eq_cons = {'type': 'eq',
                   'fun': cons}
        Gtheta.append(gtheta(para_0))
        Para_t = minimize(gtheta, para_0, constraints=eq_cons)
        para_t = Para_t.x
        mu_t = para_t[0]
        alpha_t = para_t[range(1, n+1)]
        beta_t = para_t[range(n+1, n+m+1)]
        C_t = para_t[range(n+m+1, n+m+N_s+1)]
        ltheta_new = likelihood(mu_t, alpha_t, beta_t, C_t, P_t)
        Ltheta.append(ltheta_new)
        diff = np.abs(ltheta_new - ltheta_t)
        ltheta_t = ltheta_new
    print("rep", rep)
    print("ltheta", Ltheta)
    hat_mu = mu_t
    print("mu_0", bmu)
    print("mu_t", hat_mu)
    hat_alpha = alpha_t
    hat_beta = beta_t
    hat_P = P_t
    print("C_inter", C_inter)
    print("C_t", C_t)
    hat_C = C_t
    print("prob", prob)
    print("P_t", P_t)
    result = dict(zip(["hat_mu", "hat_alpha", "hat_beta", "hat_P", "hat_C"],
                      [hat_mu, hat_alpha, hat_beta, hat_P, hat_C]))
    return result


if __name__ == '__main__':
    core = multiprocessing.cpu_count()
    initializer()
    p = Pool(core, initializer, ())
    N_rep = 1
    for rep in range(N_rep):
        Sim_results = p.apply_async(algorithm, args=(rep,), callback=log_result)
        Sim_results.get()
    p.close()
    p.join()
    mu_hat = []
    mu_bias = []
    alpha_hat = []
    alpha_bias = []
    beta_hat = []
    beta_bias = []
    C_hat = []
    C_bias = []
    Prob_hat = []
    Prob_bias = []
    for a in result_list:
        mu_hat.append(a["hat_mu"])
        mu_bias.append(a["hat_mu"]-bmu)
        alpha_hat.append(a["hat_alpha"])
        alpha_bias.append(a["hat_alpha"]-alpha)
        beta_hat.append(a["hat_beta"])
        beta_bias.append(a["hat_beta"] - beta)
        C_hat.append(a["hat_C"])
        C_bias.append(a["hat_C"]-C_inter)
        Prob_hat.append(a["hat_P"])
        Prob_bias.append(a["hat_P"]-prob)
    names = [r'$\theta^0$', r'mean$(\hat\theta)$', r'median$(\hat\theta)$', r'sd$(\hat\btheta)$',
     'mean of bias', 'rmse']
    muhat_mean = np.mean(mu_hat)
    muhat_median = np.median(mu_hat)
    muhat_sd = np.std(mu_hat)
    mu_biasmean = np.mean(np.array(mu_bias))
    mu_rmse = np.sqrt(np.mean(np.array(mu_bias)**2))
    mu_est = [bmu, muhat_mean, muhat_median, muhat_sd, mu_biasmean, mu_rmse]
    data_mu = dict(zip(names, mu_est))
    alphahat_mean = np.mean(np.array(alpha_hat), axis=0)
    alphahat_median = np.median(np.array(alpha_hat), axis=0)
    alphahat_sd = np.std(np.array(alpha_hat), axis=0)
    alpha_rmse = np.sqrt(np.mean(np.array(alpha_bias) ** 2, axis=0))
    betahat_mean = np.mean(np.array(beta_hat), axis=0)
    betahat_median = np.median(np.array(beta_hat), axis=0)
    betahat_sd = np.std(np.array(beta_hat), axis=0)
    beta_rmse = np.sqrt(np.mean(np.array(beta_bias) ** 2, axis=0))
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
    r_est = pd.DataFrame(columns=[r'$\theta^0$', r'mean$(\hat\theta)$', r'median$(\hat\theta)$', r'sd$(\hat\btheta)$',
                                     'mean of bias', 'rmse'])
    r_est = r_est.append(data_mu, ignore_index=True)
    data_C.extend(data_P)
    for g in data_C:
        r_est = r_est.append(g, ignore_index=True)
    r_est.to_csv("./groupGLM_fulldata_fulllike_data_n={}_m={}_settings.csv".format(n, m), index=True, header=True)
    plt.subplot(211)
    plt.title(r"$\hat\alpha$_bias_box_n={}_m={}".format(n, m))
    plt.boxplot(pd.DataFrame(alpha_bias).T, showfliers=False, showmeans=True)
    plt.xticks([])
    plt.subplot(212)
    plt.title(r"$\alpha^0$ and $\hat\alpha$ mean and median_n={}_m={}".format(n, m))
    plt.plot(range(n), alpha, color='black', label="alpha_true")
    plt.plot(range(n), alphahat_mean, color='blue', label="alpha_hat_mean")
    plt.plot(range(n), alphahat_median, color='green', label="alpha_hat_median")
    plt.xticks([])
    plt.savefig('./groupGLM_fulldata_fulllike_alpha_est_n={}_m={}'.format(n, m))
    plt.show()
    plt.close()
    plt.subplot(211)
    plt.title(r"$\hat\beta$_bias_box_n={}_m={}".format(n, m))
    plt.boxplot(pd.DataFrame(beta_bias).T, showfliers=False, showmeans=True)
    plt.xticks([])
    plt.subplot(212)
    plt.title(r"$\beta^0$ and $\hat\beta$ mean and median_n={}_m={}".format(n, m))
    plt.plot(range(m), beta, color='black', label="beta_true")
    plt.plot(range(m), betahat_mean, color='blue', label="beta_hat_mean")
    plt.plot(range(m), betahat_median, color='green', label="beta_hat_median")
    plt.xticks([])
    plt.savefig('./groupGLM_fulldata_fulllike_beta_est_n={}_m={}'.format(n, m))
    plt.show()
    plt.close()
    pd.DataFrame(alphahat_sd).to_csv("./groupGLM_fulldata_fulllike_alphahatsd_n={}_m={}.csv".format(n, m), index=True, header=True)
    pd.DataFrame(alpha_rmse).to_csv("./groupGLM_fulldata_fulllike_alpharmse_n={}_m={}.csv".format(n, m), index=True,
                                     header=True)
    pd.DataFrame(betahat_sd).to_csv("./groupGLM_fulldata_fulllike_betahatsd_n={}_m={}.csv".format(n, m), index=True, header=True)
    pd.DataFrame(beta_rmse).to_csv("./groupGLM_fulldata_fulllike_betarmse_n={}_m={}.csv".format(n, m), index=True,
                                     header=True)

