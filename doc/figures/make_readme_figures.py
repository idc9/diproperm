from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

from diproperm.DiProPerm import DiProPerm

X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2,
                  random_state=20)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig('data.png')
plt.close()

dpp = DiProPerm(B=1000).fit(X, y)

plt.figure(figsize=[12, 5])

# show histogram of separation statistics
plt.subplot(1, 2, 1)
dpp.plot_observed_scores()

# the observed scores
plt.subplot(1, 2, 2)
dpp.plot_perm_sep_stats(stat='md')

plt.savefig('dpp_plots.png')
plt.close()
