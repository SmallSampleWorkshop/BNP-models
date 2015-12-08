__author__ = 'jcapde87'

from syntheticdata import generate
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wishart
from scipy.special import digamma
import math


def inference(it, x, ao, Wo, vo, mo, bo, K):
    # Variational Inference for the Multivariate Gaussian Mixture Model
    N = len(x)
    ndim = x.shape[1]

    #Random initialization
    theta_I = np.random.dirichlet([ao]*K, None)
    rnk = np.random.multinomial(1, theta_I, N)
    nk  = np.matrix(np.sum(rnk,axis=0))

    # K gaussian components
    delta_I = wishart(df=vo,scale=Wo).rvs(K)
    mk = np.matrix(np.zeros((K,ndim)))
    for k in xrange(K):
        mk[k,:] = np.random.multivariate_normal(np.array(mo)[0,:],np.linalg.inv(bo*delta_I[k,:,:]),1)

    Wk =  np.zeros((K, ndim, ndim))
    Sk =  np.zeros((K, ndim, ndim))

    for i in xrange(it):
        print i
        ak  = ao + nk
        bk  = bo + nk
        vk  = vo + nk
        xk_ = np.diagflat(1./nk)*np.transpose(rnk)*x
        mk  = np.multiply(np.tile(np.transpose(1./bk),(1,ndim)),np.tile(bo*mo,(K,1))+np.diagflat(nk)*xk_)

        for k in xrange(K):
            Sk[k, :, :] = 1./nk[0,k]*np.transpose(np.diagflat(rnk[:,k])*(x-xk_[k, :]))*(x-xk_[k, :])
            inv_mat = np.linalg.inv(Wo) + nk[0,k]*np.matrix(Sk[k, :, :]) + bo*nk[0,k]/(bo+nk[0,k])*np.transpose(xk_[k, :]-mo)*(xk_[k, :]-mo)
            Wk[k, :, :] = np.linalg.inv(inv_mat)

        firstterm  = -(np.tile(ndim*1./(2*bk),(N,1)))
        secondterm = np.zeros((N,K))
        aux = np.matrix(np.zeros(K))
        for k in xrange(K):
            secondterm[:,k] = np.diag(-0.5*(vk[0,k]*(x-mk[k,:])*np.matrix(Wk[k,:,:])*np.transpose(x-mk[k,:])))
            aux[0,k] =  np.sum(digamma((vk[0, k]+1-range(ndim))/2))+ndim*math.log(2)+math.log(np.linalg.det(Wk[k, :, :]))/2

        lambdak =  np.tile(aux,(N,1))
        pik = np.tile(digamma(ak)-digamma(K*ak),(N,1))

        pnk = np.exp(pik+lambdak+firstterm+secondterm)
        rnk = np.nan_to_num(pnk/np.tile(np.sum(pnk, axis=1), (1, K)))
        nk  = np.sum(rnk, axis=0)
    z_est = np.array(np.transpose(np.argmax(rnk,axis=1))*1.)[0,]

    return z_est


if __name__ == "__main__":
    K = 5
    alpha = 1.
    So = np.array([[5, 1], [1, 5]])
    nuo = 10
    muo = np.array([[1, 0]])
    kappao = 0.01
    N = 1000

    x, z = generate(alpha, K, N, So, nuo, muo[0], kappao)

    it = 50
    alpha = 10000.
    zest = inference(it, x, alpha, So, nuo, muo, kappao, K)

    plt.subplot(121)
    plt.scatter(x[:,0],x[:,1],c=z)

    plt.subplot(122)
    plt.scatter(x[:,0],x[:,1],c=zest)
    plt.show()