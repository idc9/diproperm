from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

from diproperm.DiProPerm import DiProPerm

X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2,
                  random_state=20)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig('data.png')
plt.close()

dpp = DiProPerm(B=1000, stat='md', clf='md')
dpp.fit(X, y)

dpp.hist('md')
plt.savefig('dpp_hist.png')
plt.close()
