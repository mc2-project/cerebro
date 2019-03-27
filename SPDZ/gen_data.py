import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import json
import pickle
import sys


NUM_PARTIES = 4

def make_data(n=100000, d=10, noise=0.2, sparsity=0.2, seed=40):
	np.random.seed(seed)
	x = np.random.randn(n,d)
	w = np.random.randn(d) * (np.random.rand(d) > sparsity)
	x = np.random.randn(n,d)
	y = x @ w + np.random.randn() * noise
	return (w, x, y)



	

def main():
	n = int(sys.argv[1])
	d = int(sys.argv[2])
	seed = 50
	(w, x, y) = make_data(n * NUM_PARTIES, d, seed=seed)
	for i in range(0, NUM_PARTIES):
		d_ = {"x": x[i * n:(i+1)*n].flatten().tolist(), "y":y[i*n:(i+1)*n].flatten().tolist(), "w":w.flatten().tolist()}
		with open('data' + str(i+1) + '.json', 'w') as outfile:
			json.dump(d_, outfile)



if __name__ == "__main__":
	main()
