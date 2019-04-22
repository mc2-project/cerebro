import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math


fp = 40
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
BATCH_SIZE = 128
SGD_ITERS = len(train) / BATCH_SIZE

def sigmoid(x):
    ret = []
    for xi in x:
        denom = 1 + abs(xi[0])
        ret.append(xi[0] * 1.0 / (denom))
    ret = np.array(ret).reshape(len(ret), 1)
    return ret

w = np.zeros((dim, 1))
alpha_B = (0.01 / BATCH_SIZE)
print("alpha_B = {}".format(alpha_B))

XB = np.zeros((BATCH_SIZE, dim))
yB = np.zeros((BATCH_SIZE, 1))

for i in range(SGD_ITERS):
    batch_low = i * BATCH_SIZE
    batch_high = (i + 1) * BATCH_SIZE

    for j in range(BATCH_SIZE):
        for d in range(dim):
            XB[j][d] = X_[batch_low + j][d]
        yB[j][0] = y_[batch_low + j][0]

    w_ret = np.matmul(XB, w)
    #print("w_ret = {}".format(w_ret))
    w_sigmoid = sigmoid(w_ret)
    #print("w_sigmoid = {}".format(w_sigmoid))
    w_sub = (w_sigmoid - yB)
    #print("w_sub = {}".format(w_sub))
    XB_T = XB.T
    w_1 = np.matmul(XB_T, w_sub)
    #print("w_1 = {}".format(w_1))
    w_2 = alpha_B * w_1
    w = w - w_2
    #print("w = {}".format(w))
    
print("Finished training")

total = 0
score = 0
# Score the training
for xi, yi in zip(test_X, test_y):
    y_prob = np.dot(xi, w)
    pred = int(y_prob[0] > 0.5)
    total += 1
    score += int(pred == yi)

print(score)

print("Score = {}".format(score * 1.0 / total))
print("Weights: {}".format(w))

# Write out the training data
data_x = [v for v in X_.flatten()]
data_y = [v for v in y_.flatten()]

data = data_x + data_y
print(len(data))
data = [i * pow(2, fp) for i in data]
