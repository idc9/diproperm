import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.nearest_centroid import NearestCentroid


def get_training_fun(clf, param_grid=None, metric='roc_auc', n_splits=5):
    """
    Parameters
    ----------
    clf:
        linear classifier object following sklearn's API.

    param_grid: dict, None
        Parameters to cross-validate over. If None, will not perform cross-validation.

    metric: str
        Metric to use to select hyper-parameters. See scoring argument to GridSearchCV()

    n_splits: int
        Number of folds for cross validation

    Output
    ------
    get_scores: callable
        Takes X, y input, trains the linear classifier, extracts the normal vector
        then returns the training data scores.

    """

    def get_scores(X, y):
        """
        Selects hyper-parameters using cross-validation. Then refits
        using the full data set and returns the classification scores.

        Parameters
        -----------
        X (ndarray): dataset with observations on rows

        y (list): class labels

        Output
        -----
        let w = normalized classification vector; returns the
        training data scores (i.e. projection onto normal vector)
        which are given by s = Xw
        """

        if param_grid is None or len(param_grid) == 0:

            cv = GridSearchCV(estimator=clf,
                              param_grid=param_grid,
                              scoring=metric,
                              cv=StratifiedKFold(n_splits=n_splits),
                              refit=True).fit(X, y)

            clf_trained = cv.best_estimator_
        else:
            clf_trained = clf.fit(X, y)

        w = get_clf_normal_vector(clf_trained).reshape(-1)
        w /= np.linalg.norm(w)
        return np.dot(X, w)
    return get_scores


def get_clf_normal_vector(clf):
    """
    Returns the normal vector from a linear classifier object.

    Parameters
    ----------
    clf:
        Linear classifier object.
    """
    if hasattr(clf, 'coef_'):
        return clf.coef_

    elif type(clf) == GaussianNB:
        return get_GNB_direction(clf)

    elif type(clf) == NearestCentroid:
        return get_NC_direction(clf)

    else:
        return None


def get_NC_direction(clf):
    """
    Returns the normal vector for NearestCentroid with binary classes
    """
    # TODO:
    assert type(clf) == NearestCentroid
    assert clf.centroids_.shape[0] == 2
    return (clf.centroids_[0, :] - clf.centroids_[1, :]).reshape(-1)


def get_GNB_direction(clf):
    """
    Returns the normal vector for GaussianNB with binary classes
    """
    assert type(clf) == GaussianNB
    assert clf.theta_.shape[0] == 2
    w_md = (clf.theta_[0, :] - clf.theta_[1, :]).reshape(-1)
    p0 = clf.class_prior_[0]
    p1 = clf.class_prior_[1]
    sigma = p0 * clf.sigma_[0, :] + p1 * clf.sigma_[1, :]  # TODO: double check
    Sigma_inv = np.diag(1.0/sigma)  # TODO: safe invert
    w_nb = Sigma_inv.dot(w_md)
    return w_nb.reshape(-1)
