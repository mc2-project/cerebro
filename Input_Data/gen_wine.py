import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


float_precision = 32
rho = 0.05
local_compute = False
NUM_PARTIES = 6
num_samples_per_party = 200
def wine_dataset(nparties):
    df_r = pd.read_csv("winequality-red.csv", header=None, delimiter=";")
    df_w = pd.read_csv("winequality-white.csv", header=None, delimiter=";")
    print("Red:", df_r.shape)
    print("White:", df_w.shape)
    df = pd.concat([df_r, df_w], ignore_index=True)
    df.drop(df.index[df_r.shape[0]], inplace=True)
    df.drop(df.index[0],inplace=True)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train = train[:num_samples_per_party * NUM_PARTIES]
    values = list(train.columns.values)
    scalerX = preprocessing.StandardScaler()
    scalerY = preprocessing.StandardScaler()
    train_X = np.array(train[values[0:-1]], dtype='float32')
    train_y = np.array(train[values[-1:]], dtype='float32')
    
    scalerX.fit(train_X)
    scalerY.fit(train_y)
    train_X = scalerX.transform(train_X)
    train_y = scalerY.transform(train_y)
    
    
    test_y = np.array(test[values[-1:]], dtype='float32')
    test_X = np.array(test[values[0:-1]], dtype='float32')
    
    test_X = scalerX.transform(test_X)
    test_y = scalerY.transform(test_y)
    
    print(len(train), len(test))    
    
    
    
    
    x_data_list = []
    y_data_list = []
    samples_per_party = int(len(train) / nparties)
    
    for i in range(nparties):
        #party_i = train[samples_per_party*i:samples_per_party * (i+1)]
        #train_X = np.array(party_i[values[0:-1]], dtype='float32')
        #train_y = np.array(party_i[values[-1:]], dtype='float32')
        train_X_i = train_X[samples_per_party*i:samples_per_party * (i+1)]
        train_y_i = train_y[samples_per_party*i:samples_per_party * (i+1)]
        x_data_list.append(train_X_i)
        y_data_list.append(train_y_i)
    
    return x_data_list, y_data_list, test_X, test_y

    
    
    
    
    x_data_list = []
    y_data_list = []
    samples_per_party = int(len(train) / nparties)
    
    for i in range(nparties):
        #party_i = train[samples_per_party*i:samples_per_party * (i+1)]
        #train_X = np.array(party_i[values[0:-1]], dtype='float32')
        #train_y = np.array(party_i[values[-1:]], dtype='float32')
        train_X_i = train_X[samples_per_party*i:samples_per_party * (i+1)]
        train_y_i = train_y[samples_per_party*i:samples_per_party * (i+1)]
        x_data_list.append(train_X_i)
        y_data_list.append(train_y_i)
    
    return x_data_list, y_data_list, test_X, test_y



x_data_list, y_data_list, _, _ = wine_dataset(NUM_PARTIES)
print(len(y_data_list))
print(y_data_list[0].shape)
data = []

if local_compute:
    for i in range(NUM_PARTIES):
        x_i = x_data_list[i]
        y_i = y_data_list[i]
        inverse = np.linalg.inv(np.matmul(x_i.T, x_i) + rho * np.eye(x_i.shape[1]))

        XTy = np.matmul(x_i.T, y_i)
        for j in range(inverse.shape[0]):
            for k in range(inverse.shape[1]):
                data.append(inverse[j][k])

        

    for i in range(NUM_PARTIES):
        x_i = x_data_list[i]
        y_i = y_data_list[i]
        XTy = np.matmul(x_i.T, y_i)
        for j in range(XTy.shape[0]):
            data.append(XTy[j][0])



else:
    for item in x_data_list:
        for i in range(item.shape[0]):
            for j in range(item.shape[1]):
                data.append(item[i][j])

    for item in y_data_list:
        for i in range(item.shape[0]):
            data.append(item[i][0])



print(data)

data = [i * pow(2, float_precision) for i in data]
