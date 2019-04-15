import numpy as np
import json
import struct

float_precision = 32
rho = 0.01
l = 0.08
p = 340282366920938463463374607431768211507

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

data1 = get_data("data1.json")
data2 = get_data("data2.json")
data3 = get_data("data3.json")

x = []
y = []
x.append(np.array(data1["x"]).reshape((10, 10)))
x.append(np.array(data2["x"]).reshape((10, 10)))
x.append(np.array(data3["x"]).reshape((10, 10)))
y.append(np.array(data1["y"]).reshape((10, 1)))
y.append(np.array(data2["y"]).reshape((10, 1)))
y.append(np.array(data3["y"]).reshape((10, 1)))




XXinv_cache = []
Xy_cache = []
nparties = len(x)

d = x[0].shape[1]    
for i in range(nparties):
    x_data = x[i]
    y_data = y[i]
    
    XXinv = np.linalg.inv(np.matmul(x_data.T, x_data) + rho * np.identity(d))
    Xy = np.matmul(x_data.T, y_data)
    XXinv_cache.append(XXinv)
    Xy_cache.append(Xy)


data = []

for i in range(nparties):
	XXinv = XXinv_cache[i]
	for j in range(d):
		for k in range(d):
			data.append(XXinv[j][k] * pow(2, float_precision))


for i in range(nparties):
	XTy = Xy_cache[i]
	for j in range(d):
		data.append(XTy[j][0] * pow(2, float_precision))

"""
data = []
for item in x:
	for i in range(10):
		for j in range(10):
			data.append(item[i][j] * pow(2, float_precision))

for item in y:
	for i in range(10):
		data.append(item[i][0] * pow(2, float_precision))
	
"""
print data




