import numpy as np
import json
import struct

np.random.seed(42)
float_precision = 32
l = 0.05
p =  748288838313422294120286634350736906063837463248897
num_rows = 10
num_cols = 10
num_parties = 2

def make_ridge_data(n, d, noise=0.2, sparsity=0.2):
    w = np.random.randn(d) * (np.random.rand(d) > sparsity)
    x = np.random.randn(n, d)
    y = np.matmul(x, w) + np.random.randn() * noise
    return (w, x, y)

def get_data(fname):
    f = open(fname, 'r')
    data = json.loads(f.read())
    f.close()
    return data


def write_spdz(input_folder, data):
    f = open(input_folder + "/f0", 'w')
    for d in data:
        d = int(d)
        if (d < 0):
                d = p - d
        print d
        output = struct.pack("Q", d)
        f.write(output)
    f.close()


w, x, y = make_ridge_data(num_rows, num_cols)

data = []

for i in range(num_rows):
    for j in range(num_cols):
        data.append(x[i][j] * pow(2, float_precision))

for i in range(num_rows):
    
    data.append(y[i] * pow(2, float_precision))


print data
