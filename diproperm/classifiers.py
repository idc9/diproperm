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
    get_normal_vector: callable
        Takes X, y input, trains the linear classifier then returns
        the normal vector.


        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The X data.

        y: array-like, shape (n_samples, )
            Class labels.
    """

    def get_normal_vector(X, y):
        """
        Selects hyper-parameters using cross-validation. Then refits
        using the full data set and returns the classification scores.

        Parameters
        -----------
        X: array-like, shape (n_samples, n_features)
            The X data.

        y: array-like, shape (n_samples, )
            Class labels.

        Output
        -----

        w: array-like, shape (n_features, )
            Normalized classification vector.
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
        return w

    return get_normal_vector


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


def get_md_normal_vector(X, y):
    """
    Computes the mean difference normal vector, w_md, where
    w_md = mean(X_positive) - mean(X_negative)
    w_md = normalize(w_md)

    Parameters
    -----------
    X: array-like, shape (n_samples, n_features)
        The X data.

    y: array-like, shape (n_samples, )
        Class labels.

    Output
    -----

    w: array-like, shape (n_features, )
        Normalized classification vector.

    """
    y = np.array(y)
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    classes = np.unique(y)
    assert len(classes) == 2

    w = X[y == classes[0], :].mean(axis=0) - X[y == classes[1], :].mean(axis=0)
    w /= np.linalg.norm(w)
    return w
    # return np.dot(X, w)
