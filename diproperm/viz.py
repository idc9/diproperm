import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from diproperm.viz_utils import jitter


def plot_observed_scores(obs_scores, y):
    """
    Plots a histogram of the observed scores.

    Parameters
    ----------
    obs_scores: array-like, shape (n_samples, )
        The observed scores from projecting the data onto the classification
        normal vector.

    y: array-like, shape (n_samples, )
        The observed class labels.
    """
    classes = np.unique(y)
    mask0 = y == classes[0]

    sns.distplot(obs_scores[mask0], color='red', label=str(classes[0]))
    jitter(obs_scores[mask0], color='red')
    sns.distplot(obs_scores[~mask0], color='blue', label=str(classes[1]))
    jitter(obs_scores[~mask0], color='blue')
    plt.legend()
    plt.xlabel('observed scores')


def plot_perm_scores(perm_scores, perm_y, obs_y):
    """

    Parameters
    ----------
    perm_scores: array-like, shape (B, n_samples)
        The permutation scores.

    perm_scores: array-like, shape (B, n_samples)
        The permutation class labels.

    y_obs: array-like, shape (n_samples, )
        The observed class labels.

    """
    classes = np.unique(obs_y)

    obs0_perm0 = (obs_y == classes[0]) & (perm_y == classes[0])
    obs0_perm1 = (obs_y == classes[0]) & (perm_y == classes[1])
    obs1_perm0 = (obs_y == classes[1]) & (perm_y == classes[0])
    obs1_perm1 = (obs_y == classes[1]) & (perm_y == classes[1])

    perm0_color = 'red'
    perm1_color = 'blue'

    obs0_marker = 'o'
    obs0_size = 100
    obs1_marker = 'x'
    obs1_size = 150

    obs0_perm0_dist_kws = {'color': perm0_color, 'hist_kws': {'alpha': .1}}
    obs0_perm0_jitter_kws = {'color': perm0_color, 's': obs0_size, 'marker': obs0_marker}

    obs0_perm1_dist_kws = {'color': perm1_color, 'hist_kws': {'alpha': .1}}
    obs0_perm1_jitter_kws = {'color': perm1_color, 's': obs0_size, 'marker': obs0_marker}

    obs1_perm0_dist_kws = {'color': perm0_color, 'hist_kws': {'alpha': .1}}
    obs1_perm0_jitter_kws = {'color': perm0_color, 's': obs1_size, 'marker': obs1_marker}

    obs1_perm1_dist_kws = {'color': perm1_color, 'hist_kws': {'alpha': .1}}
    obs1_perm1_jitter_kws = {'color': perm1_color, 's': obs1_size, 'marker': obs1_marker}

    sns.distplot(perm_scores[obs0_perm0], **obs0_perm0_dist_kws)
    jitter(perm_scores[obs0_perm0], **obs0_perm0_jitter_kws)

    sns.distplot(perm_scores[obs0_perm1], **obs0_perm1_dist_kws)
    jitter(perm_scores[obs0_perm1], **obs0_perm1_jitter_kws)

    sns.distplot(perm_scores[obs1_perm0], **obs1_perm0_dist_kws)
    jitter(perm_scores[obs1_perm0], **obs1_perm0_jitter_kws)

    sns.distplot(perm_scores[obs1_perm1], **obs1_perm1_dist_kws)
    jitter(perm_scores[obs1_perm1], **obs1_perm1_jitter_kws)

    plt.xlabel('permutation scores')

    plt.legend()


def plot_perm_sep_stats(obs, perm, stat, rejected, p, Z, cuttoff, alpha, B, bins=30):
    """
    Plots a histogram of the DiProPerm distribution.

    Parameters
    ----------
    stat: str
        Which summary statistic to show.

    bins: int
        Number of bins for histogram.

    stat: str
        Name of the separation statistic.

    rejected: bool

    p: float

    Z: float

    cuttoff: float

    alpha: float

    bins: int

    """

    plt.hist(perm,
             color='black',
             label='permutation stats (B = {})'.format(B),
             bins=bins)

    jitter(perm, color='green')

    if rejected:
        obs_lw = 3
        sig_lab = 'significant'

    else:
        obs_lw = 1
        sig_lab = 'not significant'

    obs_label = 'obs stat ({}, p = {:1.3f}, Z = {:1.3f})'.\
                format(sig_lab, p, Z)

    plt.axvline(obs,
                color='red', lw=obs_lw,
                label=obs_label)

    plt.axvline(cuttoff, color='grey',
                ls='dashed',
                label='significance cutoff (alpha = {:1.3f})'.
                      format(alpha))

    plt.xlabel('DiProPerm {}'.format(stat))
    plt.legend()
