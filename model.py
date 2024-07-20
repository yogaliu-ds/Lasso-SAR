import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


def nodewise_regression(X, alpha=0.1):
    '''
    Perform nodewise regression to obtain a (N, N) matrix.
    
    Input: 
    - X: (T, N) shape return series matrix
    
    Return: 
    - final_W: (N, N) shape matrix
    '''
    T, N = X.shape
    W = np.zeros((N, N))

    temp_list = []
    for i in range(N):
        y_i = X[:, i]
        X_i = np.delete(X, i, axis=1)
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_i, y_i)
        coefficients = lasso.coef_
        intercept = lasso.intercept_
        coefficients = np.insert(coefficients, i, 0)
        temp_list.append(coefficients)
    W = np.array(temp_list)

    # Make W symmetric
    final_W = (W + W.T) / 2

    return final_W
