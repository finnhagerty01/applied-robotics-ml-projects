import numpy as np
import pandas as pd
from src.maths.linalg import log_gaussian
from src.maths.standardize import stand_pipeline
from src.maths.standardize import inverse_transform

eps_cov = 1e-10
tol = 1e-4
max_iter = 200
restarts = 5

#Full gmm loop, with E-M
def fit_gmm(data, K, max_iter = max_iter, tol = tol, eps = eps_cov, restarts = restarts, rng = None, diagonal = False):
    if rng == None:
        seed = 42
    elif isinstance(rng, int):
        seed = rng
    else:
        seed = 0

    max_ll = -np.inf
    for i in range(restarts):
        restart_rng = np.random.default_rng(seed + i)

        weights, means, covs = init_params_kmeans(data, K, restart_rng)
        
        weights, means, covs, resp, loglik_trace, final_loglik = em_loop(data, weights, means, covs, max_iter, tol, eps, diagonal)

        if final_loglik > max_ll:
            max_ll = final_loglik
            _weights, _means, _covs, _resp, _loglik_trace, _final_loglik = weights, means, covs, resp, loglik_trace, final_loglik

    BIC = compute_bic(final_loglik, data.shape[0], K, data.shape[1], diagonal)
    results = {'weights': _weights, 'means': _means, 'covs': _covs, 'resp': _resp, 'll_t': _loglik_trace, 'final_ll': _final_loglik, 'BIC': BIC}

    return results

#EM loop to get weights.
def em_loop(data, weights, means, covs, max_iter, tol, eps, diagonal):
    N = data.shape[0]
    K = len(weights)

    loglik_trace = []
    prev_ll = -np.inf
    
    for i in range(max_iter):
        resp = e_step(data, weights, means, covs, eps)

        weights, means, covs = m_step(data, resp, eps, diagonal)

        ll = compute_loglik(data, weights, means, covs, eps)
        loglik_trace.append(ll)

        if np.abs(ll-prev_ll) < tol:
            break
        
        prev_ll = ll

    final_loglik = ll

    return weights, means, covs, resp, loglik_trace, final_loglik

#Expectation step of E-M
def e_step(data, weights, means, covs, eps):
    N = data.shape[0]
    K = len(weights)

    log_resp = np.zeros((N,K))

    for k in range(K):
        log_resp[:, k] = np.log(weights[k]) + log_gaussian(data, means[k], covs[k], eps)

    max_log = np.max(log_resp, axis = 1, keepdims = True)
    log_resp_norm = log_resp - max_log
    resp = np.exp(log_resp_norm)
    resp_sum = np.sum(resp, axis = 1, keepdims=True)
    resp = resp/(resp_sum)

    return resp

#Maximization step
def m_step(data, resp, eps, diagonal):
    N, D = data.shape
    K = resp.shape[1]

    N_k = np.sum(resp, axis=0)
    weights = N_k/N

    means = []
    covs = []
    for k in range(K):
        mean_k = np.sum(resp[:, k:k+1] * data, axis = 0)/(N_k[k])
        means.append(mean_k)

    for k in range(K):
        diff = data - means[k]
        if diagonal:
            cov_k = np.sum(resp[:, k:k+1] * diff**2, axis = 0)/ (N_k[k])
            cov_k = np.diag(cov_k)
        else:
            weighted_diff = resp[:, k:k+1] * diff
            cov_k = (np.matmul(weighted_diff.T, diff)) / (N_k[k])

        cov_k = cov_k + eps * np.eye(D)
        covs.append(cov_k)
    return weights, means, covs

#Just get the log-likelihood
def compute_loglik(data, weights, means, covs, eps):
    N = data.shape[0]
    K = len(weights)

    log_prob = np.zeros((N, K))

    for k in range(K):
        log_prob[:, k] = np.log(weights[k]) + log_gaussian(data, means[k], covs[k], eps)
    
    max_log = np.max(log_prob, axis = 1)
    ll = np.sum(max_log + np.log(np.sum(np.exp(log_prob - max_log[:, np.newaxis]), axis = 1)))

    return ll

#Try and get decent initial guess with kmeans so model converges quicker
def init_params_kmeans(data, K, rng):
    N, D = data.shape

    index = rng.choice(N, K, replace = False)
    means = [data[i] for i in index]

    for i in range(10):
        dist = np.zeros((N, K))
        for k in range(K):
            diff = data - means[k]
            dist[:, k] = np.sum(diff**2, axis = 1)
        assign = np.argmin(dist, axis = 1)

        for k in range(K):
            clust_points = data[assign == k]
            if len(clust_points) > 0:
                means[k] = np.mean(clust_points, axis = 0)
    
    covs = []
    for k in range(K):
        clust_points = data[assign == k]
        if len(clust_points) > 1:
            cov_k = np.cov(clust_points.T)
        else:
            cov_k = np.eye(D)
        covs.append(cov_k)
    
    weights = np.ones(K) / K

    return weights, means, covs

#Compute the BIC
def compute_bic(loglik, n_samples, n_comps, n_feats, diagonal):
    if diagonal:
        n_params = n_comps * n_feats * 2 + n_comps - 1
    else:
        n_params = n_comps * (n_feats + n_feats * (n_feats + 1) / 2)

    bic = -2 * loglik + n_params * np.log(n_samples)

    return bic