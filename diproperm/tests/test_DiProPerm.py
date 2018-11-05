import unittest
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
import numpy as np
from statsmodels.robust.scale import mad

from diproperm.DiProPerm import DiProPerm
from diproperm.classifiers import get_training_fun


class TestRun(unittest.TestCase):

    @classmethod
    def setUp(self):
        X, y = make_blobs(n_samples=100, n_features=2, centers=2,
                          cluster_std=2, random_state=20)

        dpp = DiProPerm(B=1000, separation_stats=['md', 't', 'auc'], clf='md')
        dpp.fit(X, y)

        self.dpp = dpp
        self.X = X
        self.y = y

    def test_has_attributes(self):
        """
        Check dpp object has correct attributes
        """
        self.assertTrue(hasattr(self.dpp, 'test_stats_'))
        self.assertTrue(hasattr(self.dpp, 'obs_scores_'))
        self.assertTrue(hasattr(self.dpp, 'perm_scores_'))
        self.assertTrue(hasattr(self.dpp, 'perm_y_'))
        self.assertTrue(hasattr(self.dpp, 'obs_sep_stats_'))
        self.assertTrue(hasattr(self.dpp, 'perm_sep_stats_'))
        self.assertTrue(hasattr(self.dpp, 'obs_y_'))
        self.assertTrue(hasattr(self.dpp, 'classes_'))

    def test_keys(self):
        """
        Make sure attributes have the desired keys.
        """
        self.assertEqual(set(self.dpp.test_stats_.keys()),
                         set(['md', 't', 'auc']))

        self.assertEqual(set(self.dpp.obs_sep_stats_.keys()),
                         set(['md', 't', 'auc']))
        self.assertEqual(set(self.dpp.perm_sep_stats_.keys()),
                         set(['md', 't', 'auc']))

    def test_plotting(self):
        """
        Make sure plotting functions run.
        """

        self.dpp.plot_perm_sep_stats(stat='md')
        self.dpp.plot_observed_scores()
        self.dpp.plot_perm_scores(b=0)

    def test_custom_classifier(self):
        """
        User provided custom classifier.
        """

        clf = get_training_fun(clf=LinearSVC(max_iter=10000),
                               param_grid={'C': [.0001, .001, .01, 1, 10, 100]},
                               metric='roc_auc', n_splits=5)

        dpp = DiProPerm(B=1000, clf=clf)
        dpp.fit(self.X, self.y)

        self.assertTrue(hasattr(dpp, 'test_stats_'))

    def test_parallel(self):
        """
        Make sure the parallel processing permutation scores runs.
        """
        dpp = DiProPerm(n_jobs=-1).fit(self.X, self.y)
        self.assertTrue(hasattr(dpp, 'test_stats_'))

    def test_custom_test_stats(self):
        """
        User provided custom test statistics.
        """

        def robust_Z(obs_stat, perm_samples):
            return (obs_stat - np.median(perm_samples))/mad(perm_samples)

        custom_test_stats = [('robust_Z', robust_Z)]
        dpp = DiProPerm(custom_test_stats=custom_test_stats).fit(self.X, self.y)

        self.assertTrue('robust_Z' in dpp.test_stats_['md'].keys())

    def test_custom_sep_stats(self):
        """
        User provided custom separation statistics.
        """

        def median_difference(scores, y):
            y = np.array(y)
            classes = np.unique(y)
            assert len(classes) == 2
            s0 = scores[y == classes[0]]
            s1 = scores[y == classes[1]]
            return abs(np.median(s0) - np.median(s1))

        custom_sep_stats = [('median_difference', median_difference)]

        dpp = DiProPerm(custom_sep_stats=custom_sep_stats).fit(self.X, self.y)
        self.assertTrue('median_difference' in dpp.test_stats_.keys())
