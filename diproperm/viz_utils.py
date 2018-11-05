import numpy as np
import matplotlib.pyplot as plt


def jitter(a, jitter_base_prop=.1, jitter_spread_prop=.05,
           base=None, spread=None, ax=None, seed=None, **kwargs):

    """
    Plot datapoints in an array as points on an axis. A jitter plot is a
    scatter plot where the x position comes from the array and the y position
    is a random perturbation. Similar to a rugplot, but less
    susceptible to over-plotting.

    The jitter is a uniform sampled from a uniform on the interval
    [base - spread, base + spread]


    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.random.uniform(size=100)
    >>> plt.hist(x)
    >>> jitter(x)

    Parameters
    ----------
    a: 1D array of observations

    jitter_base_prop: float
        The base level to jitter around as a proportion of the height of the
        current plot.

    jitter_base_prop: float
        The jitter spread as a proportion of the height of the current plot.

    base: float
        The user provided base.

    spread:
        The user provided spread.

    ax: matplotlib axes, optional
         Axes to draw plot into; otherwise grabs current axes.

    seed: int
        Seed for the jitter sampling.

    kwargs:
        Key word arguments for matplotlib.pyplot.scatter
    """

    a = np.array(a)
    if ax is None:
        ax = plt.gca()
    xmin, xmax, ymin, ymax = ax.axis()

    # alias_map = dict(size="s", color="c")
    # for attr, alias in alias_map.items():
    #     if alias in kwargs:
    #         kwargs[attr] = kwargs.pop(alias)
    kwargs.setdefault("color", 'black')
    kwargs.setdefault('s', 1)

    height = ymax - ymin
    if base is None:
        base = jitter_base_prop * height + ymin
    if spread is None:
        spread = jitter_spread_prop * height

    if seed is None:
        seed = np.random.seed(seed)
    y = np.random.uniform(low=base - spread, high=base + spread, size=len(a))

    plt.scatter(a, y, zorder=2, **kwargs)
