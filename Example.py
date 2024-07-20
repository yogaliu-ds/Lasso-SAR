import numpy as np
import pandas as pd
from model import nodewise_regression

np.random.seed(0) # For reproducibility

sdf = pd.read_csv('data/SP500_Price.csv', index_col=0)
ndt = np.array(sdf)
df = 100*(np.log(ndt[1:,:])-np.log(ndt[:-1,:])) # log return * 100: to prevent the scale too small

w = nodewise_regression(df, alpha=3)

print(w)