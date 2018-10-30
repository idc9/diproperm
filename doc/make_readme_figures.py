from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

from diproperm.DiProPerm import DiProPerm

X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig('figures/data.png')
plt.close()

dpp = DiProPerm(B=1000, stat='md')
dpp.fit(X, y)

dpp.results['md']

dpp.hist('md')
plt.savefig('figures/dpp_hist.png')
plt.close()
