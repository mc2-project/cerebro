from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import math
import numpy as np
import struct
import argparse

X, y = load_breast_cancer(return_X_y =True)
X = np.array(X)
n = np.shape(X)[0]
dim = np.shape(X)[1]
y = np.array(y).reshape((n, 1))
print "n = {}, dimension = {}".format(n, dim)

# Use sklearn to train LR
clf = LogisticRegression(random_state=0, solver="lbfgs").fit(X, y)
print clf.predict(X)
#clf.predict_proba(X[:2, :])
print "sklearn score = {}".format(clf.score(X, y))

# Scale X and y
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_ = scaler_x.fit_transform(X)
y_ = scaler_y.fit_transform(y)

# Plaintext version of the secure training algorithm using SGD
BATCH_SIZE = 5
#SGD_ITERS = n / BATCH_SIZE
SGD_ITERS = 2

def sigmoid(x):
    ret = []
    for xi in x:
        denom = 1 + abs(xi[0])
        ret.append(xi[0] * 1.0 / (denom))
    ret = np.array(ret).reshape(len(ret), 1)
    return ret

w = np.zeros((dim, 1))
alpha_B = (0.01 / BATCH_SIZE)
print "alpha_B = {}".format(alpha_B)

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
    print "w_ret = {}".format(w_ret)
    w_sigmoid = sigmoid(w_ret)
    print "w_sigmoid = {}".format(w_sigmoid)
    w_sub = (w_sigmoid - yB)
    print "w_sub = {}".format(w_sub)
    XB_T = XB.T
    w_1 = np.matmul(XB_T, w_sub)
    print "w_1 = {}".format(w_1)
    w_2 = alpha_B * w_1
    w = w - w_2
    print "w = {}".format(w)
    
print "Finished training"

total = 0
score = 0
# Score the training
for xi, yi in zip(X_, y):
    y_prob = np.dot(xi, w)
    pred = int(y_prob[0] > 0.5)
    total += 1
    score += int(pred == yi)

print "Score = {}".format(score * 1.0 / total)
print "Weights: {}".format(w)

# Write out the training data
data_x = [v for v in X_.flatten()]
data_y = [v for v in y_.flatten()]

data = data_x + data_y
data = [d * (2 ** 32) for d in data]

