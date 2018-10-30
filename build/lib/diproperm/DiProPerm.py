import numpy as np
from scipy.stats import ttest_ind
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.externals.joblib import dump, load
from sklearn.metrics import roc_auc_score
from sklearn.externals.joblib import Parallel, delayed


class DiProPerm(object):
    """
    Direction-Projection-Permutation for High Dimensional Hypothesis Tests (DiProPerm)
    For details see Wei et al, 2016.

    Wei, S., Lee, C., Wichers, L., & Marron, J. S. (2016). Direction-projection-permutation for high-dimensional hypothesis tests. Journal of Computational and Graphical Statistics, 25(2), 549-569.

    Parameters
    ----------
    B: int
        Number of permutations to sample.

    clf: str, callable
        Linear classification algorithm to use. If str, must be one of ['md'].
        If callable, must take two arguments: X, y and return a vector of scores
        where the scores are Xw, w the linear classification normal vector.

    stat: str, list {'md', 't', 'auc'}
        The test statistics to compute. Multiple can be provided.

    alpha: float
        Cutoff for significance.

    n_jobs: None, int
        Number of jobs for parallel processing permutations using
        from sklearn.externals.joblib.Parallel. If None, will not use
        parallel processing.

    Attributes
    ----------

    results: dict
        dict keyed by summary statistics containing the results
        (e.g. p-value, Z statistic, etc)

    perm: dict
        dict keyed by summary statistics containing the permutation
        samples.

    metadata: dict

    classes_: list
        Class labels. For classification, classes_[0] is considered to
        be the positive class.

    """
    def __init__(self, B=100, clf='md', stat='md', alpha=0.05,
                 n_jobs=None):

        self.B = int(B)
        self.clf = clf
        self.alpha = float(alpha)
        if type(stat) != list:
            stat = [stat]
        self.stat = stat
        self.n_jobs = n_jobs

    def get_params(self):
        return {'B': self.B, 'method': self.clf,
                'stat': self.stat,
                'alpha': self.alpha, 'n_jobs': self.n_jobs}

    def __repr__(self):
        r = 'Two class DiProPerm'
        if hasattr(self, 'metadata'):
            cats = self.classes_
            r += ' of {} vs. {} \n'.format(cats[0], cats[1])
            for s in self.stat:
                r += '{}: {}\n'.format(s, self.results[s])

        return r

    def save(self, fpath, compress=9):
        dump(self, fpath, compress=compress)

    @classmethod
    def load(cls, fpath):
        return load(fpath)

    def compute_scores(self, X, y):
        if self.clf == 'md':
            return get_md_scores(X, y)
        elif callable(self.clf):
            return self.clf(X, y)
        else:
            raise ValueError("{} is invalid method. Expected: 'md' or callable")

    def get_perm_sep_stats(self, X, y):
        """
        Samples permutation separation statistics.
        """
        if self.n_jobs is not None:
            # compute permutation statistics in parallel
            ps = Parallel(n_jobs=self.n_jobs)(delayed(_get_stat)(X, y, self)
                                              for i in range(self.B))

            perm_stats = {s: np.zeros(self.B) for s in self.stat}
            for b in range(self.B):
                for s in self.stat:
                    perm_stats[s][b] = ps[b][s]
            return perm_stats

        else:
            perm_stats = {s: np.zeros(self.B) for s in self.stat}
            for b in range(self.B):
                y_perm = np.random.permutation(y)
                scores = self.compute_scores(X, y_perm)
                for s in self.stat:
                    perm_stats[s][b] = get_separation_statistic(scores, y_perm, stat=s)
            return perm_stats

    def fit(self, X, y):
        """
        X: array-like, shape (n_samples, n_features)
            The X training data matrix.

        y: array-like, shape (n_samples, )
            The observed class labels. Must be binary classes.

        """

        # check arguments
        # X, y = check_X_y(X, y)
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y)
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2

        # compute observed statistics
        obs_stat = {}
        obs_scores = self.compute_scores(X, y)
        for s in self.stat:
            obs_stat[s] = get_separation_statistic(obs_scores, y, stat=s)

        # compute permutation statistics
        perm_stats = self.get_perm_sep_stats(X, y)

        # store results
        self.perm = {}
        self.results = {}
        for s in self.stat:
            pval, Z, rejected, cutoff_val = get_test_stats(obs_stat[s],
                                                           perm_stats[s])

            self.perm[s] = {'samples': perm_stats[s],
                            'mean': np.mean(perm_stats[s]),
                            'std': np.std(perm_stats[s])}

            self.results[s] = {'obs': obs_stat[s],
                               'pval': pval,
                               'Z': Z,
                               'rejected': rejected,
                               'cutoff_val': cutoff_val}

        self.metadata = {'counter':  dict(Counter(y)),
                         'shape': X.shape}

        return self

    def hist(self, stat, bins=30):
        """
        Plots a histogram of the DiProPerm distribution.

        Parameters
        ----------
        stat: str
            Which summary statistic to show.

        bins: int
            Number of bins for histogram.
        """
        assert stat in self.results

        plt.hist(self.perm[stat]['samples'],
                 color='blue',
                 label='permutation stats',
                 bins=bins)

        if self.results[stat]['rejected']:
            obs_lw = 3
            obs_label = 'obs stat (significant, p = {})'.format(self.results[stat]['pval'])
        else:
            obs_lw = 1
            obs_label = 'obs stat (not significant, p = {})'.format(self.results[stat]['pval'])
        plt.axvline(self.results[stat]['obs'], color='red', lw=obs_lw, label=obs_label)

        plt.axvline(self.results[stat]['cutoff_val'], color='grey', ls='dashed',
                    label='significance cutoff (alpha = {})'.format(self.alpha))

        plt.xlabel('DiProPerm {} statistic'.format(stat))
        plt.legend()

    # def get_test_stats(self, stat):
    #     return {k: self.results[stat][k] for k in ['pval', 'rejected', 'Z', 'cutoff_val']}


def _get_stat(X, y, dpp):
    """
    Used for parallel processing
    """
    ps = {}
    y_perm = np.random.permutation(y)
    scores = dpp.compute_scores(X, y_perm)
    for s in dpp.stat:
        ps[s] = get_separation_statistic(scores, y_perm, stat=s)
    return ps


def get_test_stats(obs_stat, perm_stats, alpha=0.05):
    pval = np.mean(obs_stat < perm_stats)
    Z = (obs_stat - np.mean(perm_stats)) / np.std(perm_stats)

    rejected = pval < alpha
    cutoff_val = np.percentile(perm_stats, 100*(1-alpha))

    return pval, Z, rejected, cutoff_val


def get_md_scores(X, y):
    """
    Computes the mean difference scores i.e. X w_md where
    w_md = normalized( mean(X_positive) - mean(X_negative) )
    """
    y = np.array(y)
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    classes = np.unique(y)
    assert len(classes) == 2

    w = X[y == classes[0], :].mean(axis=0) - X[y == classes[1], :].mean(axis=0)
    w /= np.linalg.norm(w)
    return np.dot(X, w)


def get_separation_statistic(scores, y, stat='md'):
    y = np.array(y)
    classes = np.unique(y)
    assert len(classes) == 2
    s0 = scores[y == classes[0]]
    s1 = scores[y == classes[1]]

    if stat == 'md':
        return abs(np.mean(s0) - np.mean(s1))
    elif stat == 't':
        return abs(ttest_ind(s0, s1, equal_var=False).statistic)
    elif stat == 'auc':
        return roc_auc_score(y == classes[0], scores)
    else:
        raise ValueError("'{} is invalid statistic. Expected one of 'md' or 't'".format(stat))
