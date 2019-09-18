import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

fp = 40

# Use 1000 for the sample size. Our actually data size is 27000. So remember to multiply the results by 27x.
samples = 100
df = pd.read_excel('credit_card.xls')
df.drop(df.columns[0],axis=1,inplace=True)
# Get rid of first row which is header, list of columns
df = df[1:]

train, test = train_test_split(df, test_size=0.1, random_state=42)
train = train[:samples]
values = list(train.columns.values)
test_y = np.array(test[values[-1:]], dtype='float32')
test_X = np.array(test[values[0:-1]], dtype='float32')
train_X = np.array(train[values[0:-1]], dtype='float32')
train_y = np.array(train[values[-1:]], dtype='float32')

# Scale X and y
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_ = scaler_x.fit_transform(train_X)
y_ = scaler_y.fit_transform(train_y)

dim = np.shape(X_)[1]
# Plaintext version of the secure training algorithm using SGD
print(np.shape(X_))
# Write out the training data
data_x = [v for v in X_.flatten()]
data_y = [v for v in y_.flatten()]

data = data_x + data_y
data = [i * pow(2, fp) for i in data]

# Now we already add X and y

# The next input is the randomness provided by the parties
# Input samples * 2 random numbers
# for benchmark purpose (todo: generate real random numbers), we use 1 here.
rand_ = np.full((samples, 2), 1)
rand = [v for v in rand_.flatten()]
data = data + rand

# The next input (clear input) is the hash map. This is supposed to be hardcoded in the SCALE-MAMBA
HASH_DIMENSION = 11

# for benchmark purpose, samely, we use 1 here for the hashmap.
# The correct hash map has been generated in Input_Data/gen_hashmap.py. But we don't plan to use that for this submission.
dim = 23 # hardcoded
hashmap_ = np.full((HASH_DIMENSION, 80 + (dim + 1) * 64), 1)
hashmap = [v for v in hashmap_.flatten()]
data = data + hashmap
