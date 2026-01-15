# GMM–GMR — Learning from Robotics Data

## Goal

Apply **Gaussian Mixture Models (GMM)** and **Gaussian Mixture Regression (GMR)** to a robotics learning problem involving noisy and structured data.

---

## Key Takeaways

- Demonstrates probabilistic clustering of complex datasets
- Uses conditional inference for prediction rather than point regression
- Highlights model selection tradeoffs when choosing the number of components

---

## Example Prediction

![GMR prediction](../assets/figures/gmr_prediction.png)

---

## Learning Setup

- Inputs: measurement-derived or state-related features
- Outputs: target variables such as position or error
- Training: train/test split or cross-validation
- Metrics: RMSE, MAE, R², log-likelihood

---

## Design Decisions

- EM algorithm for GMM fitting
- Full covariance matrices
- BIC-based selection of number of mixture components
- Regularization applied to avoid singular covariances

---

## Outputs

- Log-likelihood convergence plots
- Prediction vs. ground truth comparisons
- Metric summaries

---

## How to Run

See `src/` for training and evaluation scripts.
