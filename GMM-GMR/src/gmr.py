import numpy as np
from src.maths.linalg import log_gaussian

#All the lovely math for the predict step of GMM-GMR
def gmr_predict(gmm_results, data, input_dims, output_dims, eps):
    weights = gmm_results['weights']
    means = gmm_results['means']
    covs = gmm_results['covs']

    K = len(weights)
    N = data.shape[0]
    D_out = len(output_dims)

    preds = np.zeros((N, D_out))
    vars = np.zeros((N, D_out, D_out))

    for n in range(N):
        x_in = data[n]

        h = np.zeros(K)
        for k in range(K):
            mu_in = means[k][input_dims]
            cov_k = covs[k]
            sig_inin = cov_k[np.ix_(input_dims, input_dims)]
            h[k] = weights[k] * np.exp(log_gaussian(x_in.reshape(1, -1), mu_in, sig_inin, eps))[0]

        h = h/(np.sum(h))

        y_pred = 0
        var_pred = 0
        for k in range(K):
            mu_in = means[k][input_dims]
            mu_out = means[k][output_dims]
            cov_k = covs[k]
            sig_inin = cov_k[np.ix_(input_dims, input_dims)]
            sig_outout = cov_k[np.ix_(output_dims, output_dims)]
            sig_outin = cov_k[np.ix_(output_dims, input_dims)]
            sig_inout = cov_k[np.ix_(input_dims, output_dims)]

            sig_inin_inv = np.linalg.inv(sig_inin + eps*np.eye(len(input_dims)))

            diff = x_in - mu_in
            mu_cond = mu_out + np.matmul(np.matmul(sig_outin, sig_inin_inv), diff)

            sig_cond = sig_outout - np.matmul(np.matmul(sig_outin, sig_inin_inv), sig_inout)

            y_pred = y_pred + h[k] * mu_cond
            var_pred = var_pred + h[k]**2 * sig_cond
        
        preds[n] = y_pred
        vars[n] = var_pred

    return preds, vars