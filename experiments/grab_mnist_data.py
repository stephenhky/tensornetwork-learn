
# reference: https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html?highlight=mnist

import os
import json

from tqdm import tqdm
from sklearn.datasets import fetch_openml

thisdir = os.path.dirname(__file__)

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
nbdata = X.shape[0]

f = open(os.path.join(thisdir, 'mnist_784', 'mnist_784.json'), 'w')
for i in tqdm(range(nbdata)):
    datum = {'pixels': list(X[i, :]), 'digit': y[i]}
    f.write(json.dumps(datum)+'\n')

f.close()