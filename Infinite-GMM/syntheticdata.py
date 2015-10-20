import numpy as np
from scipy.stats import invwishart
from numpy.random import multivariate_normal, dirichlet, choice


def generate(alpha, K, N, So, nuo, muo, kappao):
    d = muo.shape[0]
    x = np.zeros((N, d))
    z = np.zeros(N, dtype=int)
    sigma = []
    mu = []

    for k in xrange(K):
        sigmak = invwishart.rvs(df=nuo, scale=So)
        sigma.append(sigmak)
        muk = multivariate_normal(muo, 1/kappao*sigmak, 1)[0]
        mu.append(muk)

    pi = dirichlet(np.ones(K)*alpha)
    for i in xrange(N):
        z[i] = choice(K, 1, p=pi)[0]
        x[i, :] = multivariate_normal(mu[z[i]], sigma[z[i]], 1)[0]
    return x, z

