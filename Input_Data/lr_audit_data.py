import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

fp = 40

# Use 1000 for the sample size. Our actually data size is 27000. So remember to multiply the results by 27x.
samples = 1000
df = pd.read_excel('credit_card.xls')
print(df.shape)
print(df.columns.values)
df.drop(df.columns[0],axis=1,inplace=True)
# Get rid of first row which is header, list of columns
df = df[1:]

print(df.columns.values)

train, test = train_test_split(df, test_size=0.1, random_state=42)
print(len(train), len(test))
train = train[:samples]
values = list(train.columns.values)
test_y = np.array(test[values[-1:]], dtype='float32')
test_X = np.array(test[values[0:-1]], dtype='float32')
train_X = np.array(train[values[0:-1]], dtype='float32')
train_y = np.array(train[values[-1:]], dtype='float32')
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

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
