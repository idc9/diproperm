DiProPerm
----

**author**: `Iain Carmichael`_

Additional documentation, examples and code revisions are coming soon.
For questions, issues or feature requests please reach out to Iain:
iain@unc.edu.

Overview
========

This package implements Direction-Projection-Permutation for High Dimensional
Hypothesis Tests (DiPoPerm). For details see Wei et al, 2016 (`paper link`_, `arxiv link`_). DiProPerm "rigorously assesses whether a binary linear classifier is detecting statistically significant differences between two high-dimensional distributions."



Wei, S., Lee, C., Wichers, L., & Marron, J. S. (2016). Direction-projection-permutation for high-dimensional hypothesis tests. Journal of Computational and Graphical Statistics, 25(2), 549-569.

Installation
============
This is currently an informal package under development so I've only made it installable from github.

::

    git clone https://github.com/idc9/diproperm.git
    python setup.py install

Example
=======

.. code:: python

    from sklearn.datasets import make_blobs
    import numpy as np
    import matplotlib.pyplot as plt
    # %matplotlib inline

    from diproperm.DiProPerm import DiProPerm

    # toy binary class dataset (two isotropic Gaussians)
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2)

    # DiProPerm with mean difference classifier, mean difference summary
    # statistic, and 1000 permutation samples.
    dpp = DiProPerm(B=1000, stat='md', clf='md')
    dpp.fit(X, y)

    dpp.results['md']

.. code:: python

    {'Z': 11.704865481794599,
     'cutoff_val': 1.2678333596648679,
     'obs': 4.542253375623943,
     'pval': 0.0,
     'rejected': True}

.. code:: python

    dpp.hist('md')

.. image:: doc/figures/dpp_hist.png


For more example code see `these example notebooks`_.

Help and Support
================

Additional documentation, examples and code revisions are coming soon.
For questions, issues or feature requests please reach out to Iain:
iain@unc.edu.

Documentation
^^^^^^^^^^^^^

The source code is located on github: https://github.com/idc9/diproperm

Testing
^^^^^^^

Testing is done using `nose`.

Contributing
^^^^^^^^^^^^

We welcome contributions to make this a stronger package: data examples,
bug fixes, spelling errors, new features, etc.



.. _Iain Carmichael: https://idc9.github.io/
.. _paper link: https://www.tandfonline.com/doi/abs/10.1080/10618600.2015.1027773
.. _arxiv link: https://arxiv.org/pdf/1304.0796.pdf
.. _these example notebooks: https://github.com/idc9/diproperm/tree/master/doc
