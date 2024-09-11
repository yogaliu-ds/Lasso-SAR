# Lasso-SAR (Lasso Spatial Auto-Regressive)

Objective:
To create sparse covariance matrix, we apply node-wise L1 regulation.

# Nodewise Regression

This repository contains a Python function to perform nodewise regression using Lasso regression to estimate the relationship between nodes in a dataset. The function outputs a symmetric matrix representing the connections between the nodes.

## Code Overview

The main function in this code is nodewise_regression, which takes a matrix of return series as input and performs nodewise regression using Lasso.
nodewise_regression(X, alpha=0.1)
Parameters:

    X: A (T, N) matrix, where T is the number of time steps and N is the number of nodes (e.g., assets, variables).
    alpha: The regularization strength for Lasso regression. Default value is 0.1.

Returns:

    final_W: A symmetric (N, N) matrix representing the relationships between nodes.

Process:

    For each node i, the corresponding column of X is used as the target (y_i), and the remaining columns are used as predictors (X_i).
    Lasso regression is applied to estimate the coefficients.
    The coefficients are stored, and a symmetric weight matrix final_W is computed by averaging the coefficient matrix and its transpose.

