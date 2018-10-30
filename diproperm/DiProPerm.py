import numpy as np
from scipy.stats import ttest_ind
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.externals.joblib import dump, load
from sklearn.metrics import roc_auc_score
from sklearn.externals.joblib import Parallel, delayed

from diproperm.classifiers import get_md_scores


class DiProPerm(object):
    """
    Direction-Projection-Permutation for High Dimensional Hypothesis Tests
    (DiProPerm). See (Wei et al, 2016) for details.

    Let X1,...,Xm and Y1,...,Yn be independent random samples of d dimensional
    random vectors with multivariate distributions F1 and F2, respectively.
    We are interested interested the null hypothesis of equal distributions
    H0 : F1 = F2 versus H1 : F1 != F2.

    Wei, S., Lee, C., Wichers, L., & Marron, J. S. (2016).
    Direction-projection-permutation for high-dimensional hypothesis tests.
    Journal of Computational and Graphical Statistics, 25(2), 549-569.

    Parameters
    ----------
    B: int
        Number of permutations to sample.

    clf: str, callable
        Linear classification algorithm used to find a direction.
        If str, must be one of ['md']. If callable, must take two arguments:
        X, y and return a vector of scores where the scores are Xw, w the
        linear classification normal vector.

    stat: str, list {'md', 't', 'auc'}
        The univariate two-sample summary statistics measuring the separation
        between the two classes' scores. See section 2.2 of (Wei et al, 2016).
        Multiple can be provided as a list.

    alpha: float
        Cutoff for significance.

    custom_tests_stats: None, iterable of (str, callable) tuples
        Custom test statistics which are a function of (obs_stat, perm_samples).
        The str argument identifies the name of the statistic (used for)
        keys of test_stats attributed. The function, f, must take
        arguments f(obs_stat, perm_samples) and return a value.

    n_jobs: None, int
        Number of jobs for parallel processing permutations using
        from sklearn.externals.joblib.Parallel. If None, will not use
        parallel processing.

    Attributes
    ----------

    test_stats_: dict
        dict keyed by summary statistics containing the test statistics
        (e.g. p-value, Z statistic, etc)

    perm_samples_: dict
        dict containing the permutation statistic samples keyed by the
        summary statistics.

    metadata_: dict

    classes_: list
        Class labels. For classification, classes_[0] is considered to
        be the positive class.

    """
    def __init__(self, B=100, clf='md', stat='md', alpha=0.05,
                 custom_test_stats=None, n_jobs=None):

        self.B = int(B)
        self.clf = clf
        self.alpha = float(alpha)
        if type(stat) != list:
            stat = [stat]
        self.stat = stat
        self.custom_test_stats = custom_test_stats
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
                r += '{}: {}\n'.format(s, self.test_stats_[s])

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

            perm_samples = {s: np.zeros(self.B) for s in self.stat}
            for b in range(self.B):
                for s in self.stat:
                    perm_samples[s][b] = ps[b][s]
            return perm_samples

        else:
            perm_samples = {s: np.zeros(self.B) for s in self.stat}
            for b in range(self.B):
                y_perm = np.random.permutation(y)
                scores = self.compute_scores(X, y_perm)
                for s in self.stat:
                    perm_samples[s][b] = get_separation_statistic(scores, y_perm, stat=s)
            return perm_samples

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
        self.perm_samples_ = self.get_perm_sep_stats(X, y)

        # store results
        self.test_stats_ = {}
        for s in self.stat:
            self.test_stats_[s] = \
                get_test_statistics(obs_stat=obs_stat[s],
                                    perm_samples=self.perm_samples_[s],
                                    alpha=self.alpha,
                                    custom_test_stats=self.custom_test_stats)

        self.metadata_ = {'counter':  dict(Counter(y)),
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
        assert stat in self.test_stats_.keys()

        plt.hist(self.perm_samples_[stat],
                 color='blue',
                 label='permutation stats',
                 bins=bins)

        if self.test_stats_[stat]['rejected']:
            obs_lw = 3
            obs_label = 'obs stat (significant, p = {})'.\
                format(self.test_stats_[stat]['pval'])
        else:
            obs_lw = 1
            obs_label = 'obs stat (not significant, p = {})'.\
                format(self.test_stats_[stat]['pval'])

        plt.axvline(self.test_stats_[stat]['obs'], color='red', lw=obs_lw,
                    label=obs_label)

        plt.axvline(self.test_stats_[stat]['cutoff_val'], color='grey',
                    ls='dashed',
                    label='significance cutoff (alpha = {})'.
                          format(self.alpha))

        plt.xlabel('DiProPerm {} statistic'.format(stat))
        plt.legend()


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


def get_test_statistics(obs_stat, perm_samples, alpha=0.05,
                        custom_test_stats=None):
    """
    obs_stat: float
        The observed statistic.

    perm_samples: list
        The sampled permutation statistics.

    alpha: float, between 0 and 1
        The cutoff value.

    custom_tests_stats: None, iterable
        User provided custom test statistics. Must allow for
        for stat, f in custom_test_stats where stat is a str
        and f is a function f(obs_stat, perm_samples) returning a value.
    """

    stats = {}
    stats['obs'] = obs_stat

    stats['pval'] = np.mean(obs_stat < perm_samples)
    stats['rejected'] = stats['pval'] < alpha
    stats['cutoff_val'] = np.percentile(perm_samples, 100 * (1 - alpha))

    stats['Z'] = (obs_stat - np.mean(perm_samples)) / np.std(perm_samples)

    if custom_test_stats is not None:
        for stat, f in custom_test_stats:
            stats[stat] = f(obs_stat=obs_stat, perm_samples=perm_samples)

    return stats


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
