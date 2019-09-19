import numpy as np
import random

data_size = 27000
dim = 23
HASH_DIMENSION = 11

X = np.zeros((data_size, dim * 64))
y = np.zeros((data_size, 1 * 64))
R = np.zeros((data_size, 80))

Map = np.zeros((HASH_DIMENSION, 80 + (dim + 1) * 64))

for i in range(data_size):
    for j in range(dim * 64):
        X[i][j] = random.getrandbits(1)

for i in range(data_size):
    for j in range(1 * 64):
        y[i][j] = random.getrandbits(1)

for i in range(data_size):
    for j in range(80):
        R[i][j] = random.getrandbits(1)

for i in range(HASH_DIMENSION):
    for j in range(80 + (dim + 1) * 64):
        Map[i][j] = random.getrandbits(170) % 748288838313426946419542008036598869565988111646721

from datetime import datetime
start = datetime.now()

# We don't need the bitness check. Hooray!


RXy_bits = np.zeros((data_size, dim * 64 + 1 * 64 + 80))

for i in range(data_size):
    for j in range(dim * 64):
        RXy_bits[i][j] = X[i][j]

for i in range(data_size):
    for j in range(1 * 64):
        RXy_bits[i][dim * 64 + j] = y[i][j]

for i in range(data_size):
    for j in range(80):
        RXy_bits[i][dim * 64 + 1 * 64 + j] = R[i][j]

MapT = np.transpose(Map)

Commitments = np.matmul(RXy_bits, MapT)

print datetime.now() - start
