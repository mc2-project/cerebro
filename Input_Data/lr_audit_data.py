import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

fp = 40

# Use 100 for the sample size. Our actually data size is 27000. So remember to multiply the results by 27x.
data_size = 100

X_ = np.full((data_size, dim * 64), 1)
y_ = np.full((data_size, 64), 1)
R_ = np.full((data_size, 80), 1)

data_X = [v for v in X_.flatten()]
data_y = [v for v in y_.flatten()]
data_R = [v for v in R_.flatten()]

data_priv = data_X + data_y + data_R

# The next input (clear input) is the hash map. This is supposed to be hardcoded in the SCALE-MAMBA
HASH_DIMENSION = 11

# for benchmark purpose, samely, we use 1 here for the hashmap.
# The correct hash map has been generated in Input_Data/gen_hashmap.py. But we don't plan to use that for this submission.
dim = 23 # hardcoded
hashmap_ = np.full((HASH_DIMENSION, 80 + (dim + 1) * 64), 1)
hashmap = [v for v in hashmap_.flatten()]
data_pub = hashmap

data = [data_priv, data_pub]
