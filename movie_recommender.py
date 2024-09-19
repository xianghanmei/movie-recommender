import pandas as pd
import numpy as np
import scipy
import scipy.sparse

# Configurations
data_dir = "ml-100k/"
data_shape = (943, 1682)
df = pd.read_csv(data_dir + "u.data", sep="\t", header=None)
values = df.values