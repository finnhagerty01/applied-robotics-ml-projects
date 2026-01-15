import numpy as np
import pandas as pd
from src.load_data import gen_data, conv_dat, pare_data, split
from src.gmm_fit import fit_gmm, e_step
from src.gmr import gmr_predict
from src.visualize import plot_results, plot_robot_results, plot_evaluation_metric
from src.maths.standardize import stand_pipeline, inverse_transform, standardize_transform

def main(dataset, groups, use_real_data = False):
    if use_real_data: #Indicate what inputs and outputs are
        big_data, ground_rebased, landmark_dict = pare_data(dataset)
        
        input_cols = ['range', 'sin_bearing', 'cos_bearing', 'land_x', 'land_y',
                  'range_lag1', 'sin_bearing_lag1', 'cos_bearing_lag1', 'land_x_lag1', 'land_y_lag1',
                  'range_lag2', 'sin_bearing_lag2', 'cos_bearing_lag2', 'land_x_lag2', 'land_y_lag2']
        
        output_cols = ['x_pos', 'y_pos', 'sin_angle', 'cos_angle']
        
        full_data, inputs, outputs = conv_dat(big_data, input_cols, output_cols)
        
        K = groups
    else:
        data = gen_data(n_samples=1000, noise=0.15, seed=42)
        input_cols = ['input_1', 'input_2', 'input_3']
        output_cols = ['output_1', 'output_2']

        full_data, inputs, outputs = conv_dat(data, input_cols, output_cols)

        K = 8
    
    if use_real_data: #split data
        train_df = pd.DataFrame(full_data)
        train_data, test_data = split(train_df)
        train_data = train_data.values
        test_data = test_data.values
    else:
        train_df = pd.DataFrame(full_data)
        train_data, test_data = split(train_df)
        train_data = train_data.values
        test_data = test_data.values
    
    stand_ids = list(range(train_data.shape[1]))
    pass_ids = []
    norm_train, stats = stand_pipeline(train_data, stand_ids, pass_ids)

    gmm_results = fit_gmm(norm_train, K, diagonal=False) #Fit the model

    if use_real_data:
        input_dims = list(range(len(input_cols))) 
        output_dims = list(range(len(input_cols), len(input_cols) + len(output_cols))) 
    else:
        input_dims = [0, 1, 2]
        output_dims = [3, 4]
    eps = 1e-6

    norm_test = standardize_transform(test_data, stats)
    
    #predict the values of test data
    preds, vars = gmr_predict(gmm_results, norm_test[:, input_dims], input_dims, output_dims, eps)

    full_preds = np.hstack([norm_test[:, input_dims], preds])
    full_preds_orig = inverse_transform(full_preds, stats)
    preds_orig = full_preds_orig[:, len(input_dims):]

    destandardized_means = []
    for k in range(K):
        mean_full = np.array(gmm_results['means'][k]).reshape(1, -1)
        mean_destd = inverse_transform(mean_full, stats)
        destandardized_means.append(mean_destd[0])
    gmm_results['means_original'] = destandardized_means

    #Get eval metrics
    gmm_results['preds'] = preds_orig
    test_outputs = test_data[:, output_dims]
    mse = np.mean((test_outputs - preds_orig)**2)
    mae = np.mean(np.abs(test_outputs - preds_orig))
    rmse = np.sqrt(mse)

    
    print(f"\nTest Set Performance with {K} groupings on ds{dataset}")
    print(f"Final log-likelihood: {gmm_results['final_ll']:.2f}")
    print(f"BIC: {gmm_results['BIC']:.2f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    r2_scores = {}
    for i, col in enumerate(output_cols):
        ss_res_col = np.sum((test_outputs[:, i] - preds_orig[:, i])**2)
        ss_tot_col = np.sum((test_outputs[:, i] - np.mean(test_outputs[:, i]))**2)
        r2_col = 1 - (ss_res_col / ss_tot_col)
        r2_scores[col] = r2_col
        print(f"{col} R^2 = {r2_col:.4f}")

    gmm_results['r2'] = r2_scores

    for i, col in enumerate(output_cols):
        mse_col = np.mean((test_outputs[:, i] - preds_orig[:, i])**2)
        print(f"{col} MSE: {mse_col:.4f}")
    
    #plot results
    if use_real_data:
        train_outputs = train_data[:, output_dims]
        plot_robot_results(train_data, gmm_results, train_outputs, input_cols, output_cols, K, 'train')
        test_resp = e_step(norm_test, gmm_results['weights'], gmm_results['means'], gmm_results['covs'], 1e-6)
        gmm_results['resp'] = test_resp #for plotting purposes
        plot_robot_results(test_data, gmm_results, test_outputs, input_cols, output_cols, K, dataset)
        plot_evaluation_metric(gmm_results, test_outputs, output_cols, use_real_data)
        print(gmm_results['weights'])
        print('/n', gmm_results['resp'])
    else:
        test_resp = e_step(norm_test, gmm_results['weights'], gmm_results['means'], gmm_results['covs'], 1e-6)
        gmm_results['resp'] = test_resp #for plotting purposes
        test_indices = train_df.index[~train_df.index.isin(pd.DataFrame(train_data).index)]
        x_vals = data['x'].values[test_indices]
        plot_results(test_data, gmm_results, x_vals, test_outputs, K)
        plot_evaluation_metric(gmm_results, test_outputs, output_cols, use_real_data)
    
    return

if __name__ == '__main__':#Run with ds1, ds0 used just to explore
    # dataset = 0
    # main(dataset, 8, use_real_data=False)
    # k = np.arange(8, 13)
    # for i in k:
    #     main(dataset, i, use_real_data=True)

    dataset = 1 #Change this to test on different datasets: 0 or 1
    # for i in k: #This was for selecting the correct number of k.
    main(dataset, 11, use_real_data=True)
