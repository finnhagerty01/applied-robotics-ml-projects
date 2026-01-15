import numpy as np
import pandas as pd
import scipy.linalg as lag

#Just the log-gaussian function for the EM.
def log_gaussian(data, mu, sigma, eps = 1e-6):
    sym_cov = (sigma + sigma.T)/2
    cov_reg = sym_cov + eps * np.eye(sym_cov.shape[0])
    
    L = lag.cholesky(cov_reg, lower = True)
    r = data - mu
    y = lag.solve_triangular(L, r.T, lower = True)

    quad = np.sum(y**2, axis = 0)

    logdet = 2 * np.sum(np.log(np.diag(L)))
    logp = -.5 * (sigma.shape[0] * np.log(2*np.pi) + logdet + quad)

    return logp