import pandas as pd
import numpy as np
import scipy
import scipy.sparse
from sklearn.model_selection import train_test_split

def load_data(data_dir="ml-100k/"):
    df = pd.read_csv(data_dir + "u.data", sep="\t", header=None)
    values = df.values
    return values

def create_sparse_matrix(values, data_shape=(943, 1682)):
    M = scipy.sparse.csr_matrix((values[:, 2], (values[:, 0], values[:, 1])), dtype=np.float, shape=data_shape)
    return M

def split_data(values, test_size=0.1, random_state=2024):
    X_train, X_test = train_test_split(values, test_size=test_size, random_state=random_state)
    return X_train, X_test

if __name__ == "__main__":
    values = load_data()
    M = create_sparse_matrix(values)
    print("Sparse matrix created with shape:", M.shape)