.. _portfolio_optimisation-risk_estimators:

.. |br| raw:: html

    <br>

.. |h3| raw:: html

    <h3>

.. |h3_| raw:: html

    </h3>

.. |h4| raw:: html

    <h4>

.. |h4_| raw:: html

    </h4>

.. |h5| raw:: html

    <h5>

.. |h5_| raw:: html

    </h5>


=============
Matrix Filter
=============
Filtering a matrix from noise is an important part of data preprocessing and Feature selection when it comes to machine learning.
In this section, we will discuss a few methods that are used to filter/De-noise a matrix


.. tip::
   |h4| Underlying Literature |h4_|

   The following sources elaborate extensively on the topic:

   - **Scikit-learn User Guide on Covariance estimation** `available here <https://scikit-learn.org/stable/modules/covariance.html#robust-covariance>`__. *Describes the algorithms of covariance matrix estimators in more detail.*
   - **Minimum Downside Volatility Indices** *by* Solactive AG - German Index Engineering `available here <https://www.solactive.com/wp-content/uploads/2018/04/Solactive_Minimum-Downside-Volatility-Indices.pdf>`__. *Describes examples of use of the Semi-Covariance matrix.*
   - **Financial applications of random matrix theory: Old laces and new pieces** *by* Potter M., J.P. Bouchaud, L. Laloux `available here <https://arxiv.org/abs/physics/0507111>`__. *Describes the process of de-noising of the covariance matrix.*
   - **A Robust Estimator of the Efficient Frontier** *by* Marcos Lopez de Prado `available here <https://papers.ssrn.com/sol3/abstract_id=3469961>`__. *Describes the Constant Residual Eigenvalue Method for De-noising Covariance/Correlation Matrix.*
   - **Machine Learning for Asset Managers** *by* Marcos Lopez de Prado `available here <https://www.cambridge.org/core/books/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F38D0AE3545>`__. *Describes the Targeted Shrinkage De-noising and the De-toning methods for Covariance/Correlation Matrices.*


De-noising and De-toning Covariance/Correlation Matrix
######################################################


Spectral Clustering De-noising Method
*************************************

The main idea behind spectral clustering is to remove the noise-related eigenvalues from an empirical correlation matrix,
the method in which this is achieved is by setting the eigenvalues which are below the theoretical value to their
average value, they are set to zero in an attempt to remove the effects of those eigenvalues that are consistent
with the null hypothesis of uncorrelated random variables.

Let us consider $n$ independent random variables with finite variance and $T$ records each. Random matrix
theory allows to prove that in the $\lim\limits_{n \to \infty} T$, with a fixed ratio $Q = T/n \geq 1$, the
eigenvalues of the sample correlation matrix cannot be larger than

$$ \lambda_{max} = \sigma^2(1 + \frac{1}{Q} + 2\sqrt{\frac{1}{Q}})$$

where $\sigma^2 = 1$ for correlation matrices, once achieved we set any eignevalues above this threshold to $0$.
For example, we have a set of 5 eigenvalues sorted in the descending order ( $\lambda_1$ ... $\lambda_5$ ),
3 of which are below the maximum theoretical value, then we set

$$ \lambda_3^{NEW} = \lambda_4^{NEW} = \lambda_5^{NEW} = 0$$

- Eigenvalues above the maximum theoretical value are left intact.

$$\lambda_1^{NEW} = \lambda_1^{OLD}$$

$$\lambda_2^{NEW} = \lambda_2^{OLD}$$

- The new set of eigenvalues with the set of eigenvectors is used to obtain the new de-noised correlation matrix.
$\tilde{C}$ is the de-noised correlation matrix, $W$ is the eigenvectors matrix, and $\Lambda$ is the diagonal matrix with new eigenvalues.

$$\tilde{C} = W \Lambda W$$

- To rescale $\tilde{C}$ so that the main diagonal consists of 1s the following transformation is made. This is how the
final $C_{denoised}$ is obtained.

$$C_{denoised} = \tilde{C} [(diag[\tilde{C}])^\frac{1}{2}(diag[\tilde{C}])^{\frac{1}{2}'}]^{-1}$$

- The new correlation matrix is then transformed back to the new de-noised covariance matrix.

.. tip::

    Spectral Filtering techniques are based on the comparison between the spectrum of the sample correlation matrix and
    the spectrum expected for a random matrix. for more information about Random matrix theory check out `The process of
    de-noising the covariance matrix is described in a paper by _Potter M._, _J.P. Bouchaud_, _L. Laloux_ __“Financial
    applications of random matrix theory: Old laces and new pieces.”__  [available here](https://arxiv.org/abs/physics/0507111).


Implementation
**************

.. autoclass:: FilterMatrix
   :noindex:
   :members: denoise_covariance


----

Hierarchical Cluster Filtering  
******************************

Hierarchical Clustering, unlike K-means Clustering, does not create multiple clusters of identical size, nor does it
require a pre-defined number of clusters. Of the two different types of hierarchical clustering - Agglomerative and
Divisive - Agglomerative, or bottom-up clustering is used here.

Agglomerative Clustering assigns each observation to its own individual cluster before iteratively joining the two
most similar clusters. This process repeats until only a singular cluster remains.

Given a positive empirical correlation matrix, :math:`C` generated using :math:`n` features, the procedure given below
returns as an output a rooted tree and a filtered correlation matrix :math:`C^<` of elements :math:`c^<_{ij}`.

First, set :math:`C = C^<`. 

Then, beginning with the most highly correlated features (clusters) :math:`h` and :math:`k \in C` and the correlation
between them, :math:`c_{hk}`, one sets the elements :math:`c^<_{ij} = c^<_{ji} = c_{hk}`.

The matrix :math:`C^<` is then redefined such that:

.. math::
    \begin{cases} c^<_{qj} = f(c^<_{hj}, c^<_{kj}) & where \ j \notin h \ and \ j \notin k \\ c^<_{ij} = c^<_{ij} & otherwise \end{cases}

where :math:`f(c^<_{hj}, c^<_{kj})` is any distance metric. In effect, merging the clusters :math:`h` and :math:`k`.
These steps are then completed for the next two most similar clusters, and are repeated for a total
of :math:`n-1` iterations; until only a single cluster remains.


.. tip::
    Divisive Hierarchical clustering works in the opposite way. It starts with one single cluster wrapping all
    data points and divides the cluster at each step of its iteration until it ends with n clusters. For more information
    on Hierarchical Procedures for correlation matrix filtering, Check Michele Tumminello et al research paper
    `here <https://arxiv.org/pdf/0809.4615.pdf>`__.


Implementation
**************

.. autoclass:: FilterMatrix
   :noindex:
   :members: filter_corr_hierarchical


----

Example Code
############

.. code-block::

    import pandas as pd
    import numpy as np
    from datascience.filter.filter import FilterMatrix

    # Import returns data
    stock_returns = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)

    # Class that have needed functions
    filt = FilterMatrix()

    # Relation of number of observations T to the number of variables N (T/N)
    tn_relation = stock_prices.shape[0] / stock_prices.shape[1]

    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01

    # Series of returns from series of prices
    stock_returns = stock_prices.pct_change()
    stock_returns.dropna(inplace=True)
	
    # Finding the simple covariance matrix from a series of returns
    cov_matrix = stock_returns.cov()

    # Finding the correlation matrix from a series of returns
    corr_matrix = stock_returns.corr()

    # Finding the Spectral Clustering De-noised Сovariance matrix
    const_resid_denoised = filt.denoise_covariance(cov_matrix, tn_relation)

    # Finding the Hierarchical Clustering Filtered Correlation matrix
    hierarchical_filtered = filt.filter_corr_hierarchical(corr_matrix, method='complete',
                                                                     draw_plot=False)

Research Notebooks
##################

The following research notebook can be used to better understand how the algorithms within this module can be used on real data.

* `Risk Estimators Notebook`_

.. _Risk Estimators Notebook: https://github.com/skyliquid22/datascience/blob/master/notebooks/MatrixFilter.ipynb
