__author__ = 'jcapde87'

from syntheticdata import generate
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, multivariate_normal
from math import pi, isnan
from numpy.random import choice

def mv_tstudent_pdf(x, nu, mu, Sigma):
    #It is only valid for p=2
    p = 2
    Denom = (pow(nu*pi,1.*p/2) * pow(np.linalg.det(Sigma),1./2)
             *pow(1 + (1./nu)*np.dot(np.dot((x - mu),np.linalg.inv(Sigma)), (x - mu).T),1.* (p+nu)/2))
    Num = nu/2
    return 1. * Num / Denom

def collapsed_sampler(it, x, alpha, So, nuo, muo, kappao):
    N, d = x.shape
    K = 1
    Nk = np.array([N])
    z = np.zeros(N)
    momoT = kappao*np.dot(muo.T,muo)

    for i in xrange(it):
        for n in xrange(N):
            xn = x[n,:]
            Nk[z[n]] -= 1
            Pzn = np.zeros(K+1)
            for k in xrange(K):
                pzn_aux = Nk[k]/(N+alpha-1.)
                xk = x[np.where(z==k)[0],:]
                Sk = np.dot(xk.T,xk)
                kappa_n = kappao + Nk[k]
                nu_n = nuo + Nk[k]
                mu_n = (kappao*muo+Nk[k]*xk.mean(axis=0))/kappa_n
                m_nm_nT = kappa_n*np.dot(mu_n.T,mu_n)
                sigma_n = So + Sk + momoT - m_nm_nT
                pxn_aux = mv_tstudent_pdf(xn, nu_n-d+1, mu_n, (kappa_n+1)*sigma_n/(kappa_n*(nu_n-d+1)))
                Pzn[k] = pzn_aux * pxn_aux
            pzn_aux = alpha/(N+alpha-1.)
            pxn_aux = mv_tstudent_pdf(xn, nuo-d+1, muo, So)
            Pzn[K] = pzn_aux * pxn_aux
            Pzn_norm = Pzn/sum(Pzn)
            z[n] = choice(K+1, 1, p=Pzn_norm)[0]
            K = len(set(z))
            newk = 0
            Nk = np.zeros(K)
            for oldk in list(set(z)):
                ind = np.where(z==oldk)[0]
                z[ind] = newk
                Nk[newk] = len(ind)
                newk +=1
        print "Iteration " + str(i) + " completed! Number of clusters: " + str(len(set(z)))
    return z


if __name__ == "__main__":
    K = 5
    alpha = 1.
    So = np.array([[5, 1], [1, 5]])
    nuo = 10
    muo = np.array([[1, 0]])
    kappao = 0.01
    N = 1000

    x, z = generate(alpha, K, N, So, nuo, muo[0], kappao)
    plt.scatter(x[:,0],x[:,1],c=z)
    plt.title("Gaussian clusters")
    plt.show()

    it = 50
    alpha = 10000.
    zest = collapsed_sampler(it, x, alpha, So, nuo, muo, kappao)

    plt.scatter(x[:,0],x[:,1],c=zest)
    plt.title("Estimated Gaussian clusters")
    plt.show()