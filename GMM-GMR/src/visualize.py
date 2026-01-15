import matplotlib.pyplot as plt
import numpy as np

def plot_results(data, results, x_vals, outputs, k):
    resp = results['resp']
    means = results['means_original']
    covs = results['covs']
    weights = results['weights']

    assign = np.argmax(resp, axis = 1)
    K = len(weights)

    fig = plt.figure(figsize=(15, 6))

    ax1 = plt.subplot(1, 3, 1)
    for k in range(K):
        mask = assign == k
        ax1.scatter(data[mask, 0], data[mask, 1], alpha = .1, label = f'cluster {k}')
        ax1.scatter(means[k][0], means[k][1], marker = 'x', s = 10, c = 'red')

    ax1.set_xlabel('Input 1')
    ax1.set_ylabel('Input 2')
    ax1.set_title('Clusters of Input 1 and 2')

    ax2 = plt.subplot(1, 3, 2)
    for k in range(K):
        mask = assign == k
        ax2.scatter(data[mask, 0], data[mask, 2], alpha = .1, label = f'cluster {k}')
        ax2.scatter(means[k][0], means[k][2], marker = 'x', s = 10, c = 'red')

    ax2.set_xlabel('Input 1')
    ax2.set_ylabel('Input 3')
    ax2.set_title('Clusters of Input 1 and 3')

    ax3 = plt.subplot(1, 3, 3)
    for k in range(K):
        mask = assign == k
        ax3.scatter(data[mask, 1], data[mask, 2], alpha = .1, label = f'cluster {k}')
        ax3.scatter(means[k][1], means[k][2], marker = 'x', s = 10, c = 'red')

    ax3.set_xlabel('Input 2')
    ax3.set_ylabel('Input 3')
    ax3.set_title('Clusters of Input 2 and 3')
    plt.savefig(f'plots/synthetic_data')
    plt.show()

    fig = plt.figure(figsize=(15, 6))
    ax4 = plt.subplot(1, 2, 1)
    ax4.scatter(x_vals, outputs[:, 0], alpha = .5, label = 'True')
    ax4.scatter(x_vals, results['preds'][:, 0], alpha = .1, label = 'Predicted')
    ax4.set_xlabel('x')
    ax4.set_ylabel('Output 1')
    ax4.set_title('Output 1 Predictions')
    ax4.legend(bbox_to_anchor=(-.05, 1), loc='upper left', fontsize=8)

    ax6 = plt.subplot(1, 2, 2)
    ax6.plot(results['ll_t'])
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Log-Likelihood')
    ax6.set_title('Training Convergence')
    
    plt.savefig(f'plots/synthetic_data_results')
    plt.show()

def plot_robot_results(data, results, outputs, input_cols, output_cols, k, dataset):
    resp = results['resp']
    means = results['means_original']
    weights = results['weights']
    preds = results['preds']
    
    assign = np.argmax(resp, axis=1)
    K = len(weights)
    
    x_idx = output_cols.index('x_pos')
    y_idx = output_cols.index('y_pos')
    
    n_inputs = len(input_cols)
    output_start_idx = n_inputs
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = plt.subplot(1, 3, 1)
    for k in range(K):
        mask = assign == k
        ax1.scatter(data[mask, output_start_idx + x_idx], data[mask, output_start_idx + y_idx], alpha=0.3, s=20, label=f'Cluster {k}')
        ax1.scatter(means[k][output_start_idx + x_idx], means[k][output_start_idx + y_idx],marker='x', s=40, linewidths=3, c='red')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('GMM Clusters in Robot World')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(1, 3, 2)
    range_idx = input_cols.index('range')
    sin_idx = input_cols.index('sin_bearing')
    for k in range(K):
        mask = assign == k
        ax2.scatter(data[mask, range_idx], data[mask, sin_idx], alpha=0.3, s=20, label=f'Cluster {k}')
        ax2.scatter(means[k][range_idx], means[k][sin_idx],marker='x', s=40, linewidths=3, c='red')
    ax2.set_xlabel('Range (m)')
    ax2.set_ylabel('sin(bearing)')
    ax2.set_title('Clusters in Measurement Space')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(results['ll_t'])
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Log-Likelihood')
    ax3.set_title('Training Convergence')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/robot_data{dataset}_results_{k}.png')
    return

def plot_evaluation_metric(gmm_results, test_outputs, output_cols, real_data = False):

    r2_scores = gmm_results['r2']
    predictions = gmm_results['preds']
    
    n_outputs = len(output_cols)
    fig, axes = plt.subplots(1, n_outputs, figsize=(6*n_outputs, 5))
    
    for i, col in enumerate(output_cols):
        ax = axes[i]
        ax.scatter(test_outputs[:, i], predictions[:, i], alpha=0.4, s=20, c='blue')
        
        min_val, max_val = test_outputs[:, i].min(), test_outputs[:, i].max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel(f'True {col}')
        ax.set_ylabel(f'Predicted {col}')
        ax.set_title(f'{col}\n$R^2$ = {r2_scores[col]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    if not real_data:
        plt.savefig('plots/r2_evaluation_synth.png')
    else:
        plt.savefig('plots/r2_evaluation.png')