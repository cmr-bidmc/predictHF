import scipy.io as sio
import numpy as np
import hyper_params as hp
hp = hp.create_hparams()

def load_data(type, dataset_path, normalize=True):
    dataset = sio.loadmat(dataset_path)
    data = dataset['no_impute_xls_data'] # matlab variable name that contains the dataset
    P = data[1:, 0] # Extract Patient ID
    X = data[1:, 1:-1] # Exclude: row_0 (Feature ID) and col_0(Patient ID); and the last column (outcome)
    Y = data[1:, -1]

    for f in range(X.shape[1]):  # loop on features
        f_max = np.nanmax(X[:, f])
        f_min = np.nanmin(X[:, f])
        X[:, f] = np.divide(np.subtract(X[:, f], f_min), (f_max - f_min+ 0.0000000001))

    return X.astype(float),Y.astype(float),P

def calc_impute_values(X):
    f_impute = []
    for f in range(X.shape[1]): # loop on features
        f_impute.append(np.nanmean(X[:,f]))
    return f_impute

def impute_data(X,impute_values):
    for f in range(X.shape[1]): # loop on features
        for p in range(X.shape[0]):  # loop on patients
            if np.isnan(X[p,f]):
                X[p,f] = impute_values[f]
    return X
