import numpy as np
from scipy.stats import ttest_ind
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.externals.joblib import dump, load
from sklearn.metrics import roc_auc_score
from sklearn.externals.joblib import Parallel, delayed

from diproperm.classifiers import get_md_normal_vector
from diproperm.viz import plot_perm_sep_stats, plot_observed_scores, \
    plot_perm_scores


class DiProPerm(object):
    """
    Direction-Projection-Permutation for High Dimensional Hypothesis Tests
    (DiProPerm). See (Wei et al, 2016) for details.

    Let X1,...,Xm and Y1,...,Yn be independent random samples of d dimensional
    random vectors with multivariate distributions F1 and F2, respectively.
    We are interested interested the null hypothesis of equal distributions
    H0 : F1 = F2 versus H1 : F1 != F2.

    DiProPerm does the following procedure once for the observed data (X, y)
    and then B times for permuted data (X, y_perm).

    - fit a linear classifier normal vector, w, (e.g. using the
    mean difference or SVM).

    - project the data, X, onto the normal vector to
    obtain the scores, Xw.

    - compute a separation statistic measuring how well separated the
    two class scores are (e.g. ROC-AUC, difference of class means).

    The B permuted separation statistics are samples from the null
    distribution and can be used to conduct a hypothesis test.


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

    separation_stats: str, list {'md', 't', 'auc'}
        The univariate two-sample statistics measuring the separation
        between the two classes' scores. See section 2.2 of (Wei et al, 2016).
        Multiple can be provided as a list.


    custom_sep_stats: None, iterable of (str, callable) tuples
        Custom separation statistics which are a function of (scores, y).
        The str argument identifies the name of the statistic (used for
        keys of obs_sep_stats_, perm_sep_stats_ attributes).
        The function, f, must take arguments f(scores, y) and return a value.
        The larger the value, the more separated the two classes are.

    custom_tests_stats: None, iterable of (str, callable) tuples
        Custom test statistics which are a function of (obs_stat, perm_samples).
        The str argument identifies the name of the statistic (used for
        keys of test_stats_ attribute). The function, f, must take
        arguments f(obs_stat, perm_samples) and return a value.

    alpha: float
        Cutoff for significance.

    n_jobs: None, int
        Number of jobs for parallel processing permutations using
        sklearn.externals.joblib.Parallel. If None, will not use
        parallel processing.

    Attributes
    ----------

    test_stats_: dict
        dict keyed by summary statistics containing the test statistics
        (e.g. p-value, Z statistic, etc)

    obs_scores_: np.arrays, shape (n_samples, )
        The observed scores (i.e. projection of the training data onto
        the classification normal vector).

    perm_scores_: np.arrays, shape (B, n_samples)
        The permutation scores.

    perm_y: np.array, shape (B, n_samples)
        The permuted labels used for each permutation sample.

    obs_sep_stats_: dict of np.arrays, shape (n_samples, )
        dict containing the observed separation statistics. Keys are the
        separation statistics (e.g. 'md', 't', 'auc').

    perm_sep_stats_: dict of np.array, shape (B, n_samples)
        dict containing the permuted separation statistics.

    metadata_: dict

    obs_y_: array-like
        The observed class labels i.e. the y argument to fit

    classes_: list
        Class labels. For classification, classes_[0] is considered to
        be the positive class.

    """
    def __init__(self, B=100, clf='md',
                 separation_stats=['md', 'auc'],
                 custom_sep_stats=None,
                 custom_test_stats=None,
                 alpha=0.05, n_jobs=None):

        self.B = int(B)
        self.clf = clf
        self.n_jobs = n_jobs
        if type(separation_stats) == str:
            separation_stats = [separation_stats]
        self.separation_stats = separation_stats
        self.alpha = float(alpha)
        self.custom_sep_stats = custom_sep_stats
        self.custom_test_stats = custom_test_stats

    def get_params(self):
        return {'B': self.B, 'clf': self.clf,
                'custom_sep_stats': self.custom_sep_stats,
                'custom_test_stats': self.custom_test_stats,
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
        """
        Returns the scores of a linear classifier fit on X, y.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The X data matrix.

        y: array-list, shape (n_samples,)
            The binary class labels.
        """
        if self.clf == 'md':
            w = get_md_normal_vector(X, y)
        elif callable(self.clf):
            w = self.clf(X, y)
        else:
            raise ValueError("{} is invalid method. Expected: 'md' or callable")

        w = w.reshape(-1)
        w /= np.linalg.norm(w)
        return np.dot(X, w)

    def fit(self, X, y):
        """
        Fits DiProPerm.

        Parameters
        ----------
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
        self.obs_y_ = y
        self.metadata_ = {'counter':  dict(Counter(y)),
                          'shape': X.shape}

        self.obs_scores_ = self.compute_scores(X, y)
        self.perm_scores_, self.perm_y_ = self._compute_perm_scores(X, y)

        self.obs_sep_stats_ = {}
        self.perm_sep_stats_ = {}
        self.test_stats_ = {}
        self.compute_statistics()
        return self

    def compute_statistics(self):
        """

        Computes separation and test statistics after permutation and observed
        scores have been computed.
        """
        assert hasattr(self, 'obs_scores_') and hasattr(self, 'perm_scores_')

        for stat in self.separation_stats:
            obs_sep = get_separation_statistic(self.obs_scores_, self.obs_y_,
                                               stat=stat)

            perm_sep = np.array([get_separation_statistic(self.perm_scores_[b, :],
                                                          self.obs_y_, stat=stat)
                                 for b in range(self.B)])

            tst = get_test_statistics(obs_sep, perm_sep, alpha=self.alpha,
                                      custom_test_stats=self.custom_test_stats)

            self.obs_sep_stats_[stat] = obs_sep
            self.perm_sep_stats_[stat] = perm_sep
            self.test_stats_[stat] = tst

        # user provided separation statistics
        if self.custom_sep_stats is not None:
            for stat, f in self.custom_sep_stats:
                obs_sep = f(scores=self.obs_scores_, y=self.obs_y_)
                perm_sep = np.array([f(scores=self.perm_scores_[b, :], y=self.obs_y_)
                                     for b in range(self.B)])

                tst = get_test_statistics(obs_sep, perm_sep, alpha=self.alpha,
                                          custom_test_stats=self.custom_test_stats)

                self.obs_sep_stats_[stat] = obs_sep
                self.perm_sep_stats_[stat] = perm_sep
                self.test_stats_[stat] = tst

    def _sample_perm(self, X, y):
        """
        Samples scores from one permutation.

        Output
        ------
        perm_scores, perm_y

        perm_scores: array-like, shape (n_samples, )
            The scores computed from the permutation distribution.

        perm_y: array-like, shape (n_samples, )
            The permuted class labels.

        """
        perm_y = np.random.permutation(y)
        perm_scores = self.compute_scores(X, perm_y)
        return perm_scores, perm_y

    def _compute_perm_scores(self, X, y):
        """
        Samples permutation scores.
        """

        # compute permutation statistics in parallel
        if self.n_jobs is not None:
            perm = Parallel(n_jobs=self.n_jobs)\
                    (delayed(self._sample_perm)(X, y) for i in range(self.B))

        else:
            perm = [self._sample_perm(X, y) for _ in range(self.B)]

        perm_scores, y_perm = tuple(zip(*perm))
        return np.array(perm_scores), np.array(y_perm)

    def plot_perm_sep_stats(self, stat):
        """
        Plots a histogram of the DiProPerm null distribution with the
        observed value and some test statistics.

        Parameters
        ----------
        stat: str
            Which separation statistic to show.
        """
        assert stat in self.obs_sep_stats_.keys()

        plot_perm_sep_stats(obs=self.obs_sep_stats_[stat],
                            perm=self.perm_sep_stats_[stat],
                            stat=stat,
                            rejected=self.test_stats_[stat]['rejected'],
                            p=self.test_stats_[stat]['pval'],
                            Z=self.test_stats_[stat]['Z'],
                            cuttoff=self.test_stats_[stat]['cutoff_val'],
                            B=self.B,
                            alpha=self.alpha)

    def plot_observed_scores(self):
        """
        Plots a histogram of the observed scores for the training data.
        """
        plot_observed_scores(self.obs_scores_, self.obs_y_)

    def plot_perm_scores(self, b=0):
        """
        Plots a histogram of the permuted scores for one of the permutation
        samples.

        Parameters
        ----------
        b: int
            Index of the permutation sample to plot.
        """
        plot_perm_scores(perm_scores=self.perm_scores_[b, :],
                         perm_y=self.perm_y_[b, :], obs_y=self.obs_y_)


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
