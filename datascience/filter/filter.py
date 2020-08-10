# pylint: disable=missing-module-docstring
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from scipy.cluster.hierarchy import average, complete, single, dendrogram
from matplotlib import pyplot as plt

class Filtermatrix:

    def denoise_covariance(self, cov, tn_relation, kde_bwidth=0.01):
        """
        De-noises the covariance matrix or the correlation matrix using Spectral clustering.

        The Spectral Method works just like the Constant Residual Eigenvalue Method, but instead of replacing
        eigenvalues lower than the maximum theoretical eigenvalue to their average value, they are replaced with
        zero instead.

        Correlation matrix can also be detoned by excluding a number of first eigenvectors representing
        the market component.

        These algorithms are reproduced with minor modifications from the following book:
        Marcos Lopez de Prado “Machine Learning for Asset Managers”, (2020).

        :param cov: (np.array) Covariance matrix or correlation matrix.
        :param tn_relation: (float) Relation of sample length T to the number of variables N used to calculate the
                                    covariance matrix.
        :param kde_bwidth: (float) The bandwidth of the kernel to fit KDE.
        :return: (np.array) De-noised covariance matrix or correlation matrix.
        """

        # Correlation matrix computation (if correlation matrix given, nothing changes)

        corr = self.cov_to_corr(cov)

        # Calculating eigenvalues and eigenvectors
        eigenval, eigenvec = self._get_pca(corr)

        # Calculating the maximum eigenvalue to fit the theoretical distribution
        maximum_eigen, _ = self._find_max_eval(np.diag(eigenval), tn_relation, kde_bwidth)

        # Calculating the threshold of eigenvalues that fit the theoretical distribution
        # from our set of eigenvalues
        num_facts = eigenval.shape[0] - np.diag(eigenval)[::-1].searchsorted(maximum_eigen)

        # Based on the threshold, de-noising the correlation matrix
        corr = self._denoised_corr_spectral(eigenval, eigenvec, num_facts)

        # Calculating the covariance matrix from the de-noised correlation matrix
        cov_denoised = self.corr_to_cov(corr, np.diag(cov) ** (1 / 2))

        return cov_denoised

    def _denoised_corr_spectral(self, eigenvalues, eigenvectors, num_facts):
        """
        De-noises the correlation matrix using the Spectral method.

        The input is the eigenvalues and the eigenvectors of the correlation matrix and the number
        of the first eigenvalue that is below the maximum theoretical eigenvalue.

        De-noising is done by shrinking the eigenvalues associated with noise (the eigenvalues lower than
        the maximum theoretical eigenvalue are set to zero, preserving the trace of the
        correlation matrix).
        The result is the de-noised correlation matrix.

        :param eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal.
        :param eigenvectors: (float) Eigenvectors array.
        :param num_facts: (float) Threshold for eigenvalues to be fixed.
        :return: (np.array) De-noised correlation matrix.
        """

        # Vector of eigenvalues from the main diagonal of a matrix
        eigenval_vec = np.diag(eigenvalues).copy()

        # Replacing eigenvalues after num_facts to zero
        eigenval_vec[num_facts:] = 0

        # Back to eigenvalues on main diagonal of a matrix
        eigenvalues = np.diag(eigenval_vec)

        # De-noised correlation matrix
        corr = np.dot(eigenvectors, eigenvalues).dot(eigenvectors.T)

        # Rescaling the correlation matrix to have 1s on the main diagonal
        corr = self.cov_to_corr(corr)

        return corr

    def _find_max_eval(self, eigen_observations, tn_relation, kde_bwidth):
        """
        Searching for maximum random eigenvalue by fitting Marcenko-Pastur distribution
        to the empirical one - obtained through kernel density estimation. The fit is done by
        minimizing the Sum of Squared estimate of Errors between the theoretical pdf and the
        kernel fit. The minimization is done by adjusting the variation of the M-P distribution.

        :param eigen_observations: (np.array) Observed empirical eigenvalues. (for the empirical pdf)
        :param tn_relation: (float) Relation of sample length T to the number of variables N. (for the theoretical pdf)
        :param kde_bwidth: (float) The bandwidth of the kernel. (for the empirical pdf)
        :return: (float, float) Maximum random eigenvalue, optimal variation of the Marcenko-Pastur distribution.
        """

        # Searching for the variation of Marcenko-Pastur distribution for the best fit with the empirical distribution
        optimization = minimize(self._pdf_fit, x0=np.array(0.5), args=(eigen_observations, tn_relation, kde_bwidth),
                                bounds=((1e-5, 1 - 1e-5),))

        # The optimal solution found
        var = optimization['x'][0]

        # Eigenvalue calculated as the maximum expected eigenvalue based on the input
        maximum_eigen = var * (1 + (1 / tn_relation) ** (1 / 2)) ** 2

        return maximum_eigen, var

    def _pdf_fit(self, var, eigen_observations, tn_relation, kde_bwidth, num_points=1000):
        """
        Calculates the fit (Sum of Squared estimate of Errors) of the empirical pdf
        (kernel density estimation) to the theoretical pdf (Marcenko-Pastur distribution).

        SSE is calculated for num_points, equally spread between minimum and maximum
        expected theoretical eigenvalues.

        :param var: (float) Variance of the M-P distribution. (for the theoretical pdf)
        :param eigen_observations: (np.array) Observed empirical eigenvalues. (for the empirical pdf)
        :param tn_relation: (float) Relation of sample length T to the number of variables N. (for the theoretical pdf)
        :param kde_bwidth: (float) The bandwidth of the kernel. (for the empirical pdf)
        :param num_points: (int) Number of points to estimate pdf. (for the empirical pdf, 1000 by default)
        :return: (float) SSE between empirical pdf and theoretical pdf.
        """

        # Calculating theoretical and empirical pdf
        theoretical_pdf = self._mp_pdf(var, tn_relation, num_points)
        empirical_pdf = self._fit_kde(eigen_observations, kde_bwidth, eval_points=theoretical_pdf.index.values)

        # Fit calculation
        sse = np.sum((empirical_pdf - theoretical_pdf) ** 2)

        return sse

    @staticmethod
    def filter_corr_hierarchical(cor_matrix, method='complete', draw_plot=False):
        """
        Creates a filtered correlation matrix using hierarchical clustering methods from an empirical
        correlation matrix, given that all values are non-negative [0 ~ 1]

        This function allows for three types of hierarchical clustering - complete, single, and average
        linkage clusters. Link to hierarchical clustering methods documentation:
        `<https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_

        It works as follows:

        First, the method creates a hierarchical clustering tree using scipy's hierarchical clustering methods
        from the empirical 2-D correlation matrix.

        Second, it extracts and stores each cluster's filtered value (alpha) and assigns it to it's corresponding leaf.

        Finally, we create a new filtered matrix by assigning each of the correlations to their corresponding
        parent node's alpha value.

        :param cor_matrix: (np.array) Numpy array of an empirical correlation matrix.
        :param method: (str) Hierarchical clustering method to use. (``complete`` by default, ``single``, ``average``)
        :param draw_plot: (bool) Plots the hierarchical cluster tree. (False by default)
        :return: (np.array) The filtered correlation matrix.
        """

        # Check if all matrix elements are positive
        if np.any(cor_matrix < 0):
            warnings.warn('Not all elements in matrix are positive... Returning unfiltered matrix.', UserWarning)
            return cor_matrix

        # Check if matrix is 2-D
        if len(cor_matrix.shape) == 2:
            cor_x, cor_y = cor_matrix.shape
        else:
            warnings.warn('Invalid matrix dimensions, input must be 2-D array... Returning unfiltered matrix.',
                          UserWarning)
            return cor_matrix

        # Check if matrix dimensions and diagonal values are valid.
        if cor_x == cor_y and np.allclose(np.diag(cor_matrix), 1):  # using np.allclose as diag values might be 0.99999
            # Creating new coorelation condensed matrix for the upper triangle and dismissing the diagnol.
            new_cor = cor_matrix[np.triu_indices(cor_matrix.shape[0], k=1)]
        else:
            warnings.warn(
                'Invalid matrix, input must be a correlation matrix of size (m x m)... Returning unfiltered matrix.',
                UserWarning)
            return cor_matrix

        # Compute the hierarchical clustering tree
        if method == 'complete':
            z_cluster = complete(new_cor)
        elif method == 'single':
            z_cluster = single(new_cor)
        elif method == 'average':
            z_cluster = average(new_cor)
        else:
            warnings.warn('Invalid method selected, please check docstring... Returning unfiltered matrix.',
                          UserWarning)
            return cor_matrix

        # Plot the hierarchical cluster tree
        if draw_plot:
            fig = plt.figure(figsize=(10, 6))
            axis = fig.add_subplot(111)
            dendrogram(z_cluster, ax=axis)
            plt.show()

        # Creates a pd.DataFrame that will act as a dictionary where the index is the leaf node id, and the values are
        # thier corresponding cluster's alpha value
        alpha_values = z_cluster[:, 2]
        alphas = z_cluster[:, 0]
        df_alphas = pd.DataFrame(alpha_values, index=alphas)
        df_alphas.loc[z_cluster[0][1]] = alpha_values[0]

        # Creates the filtered correlation matrix
        alphas_sorterd = df_alphas.sort_index()
        alphas_x = np.tile(alphas_sorterd.values, (1, len(alphas_sorterd.values)))
        filt_corr = np.maximum(alphas_x, alphas_x.T)
        np.fill_diagonal(filt_corr, 1)

        return filt_corr

    @staticmethod
    def _fit_kde(observations, kde_bwidth=0.01, kde_kernel='gaussian', eval_points=None):
        """
        Fits kernel to a series of observations (in out case eigenvalues), and derives the
        probability density function of observations.

        The function used to fit kernel is KernelDensity from sklearn.neighbors. Fit of the KDE
        can be evaluated on a given set of points, passed as eval_points variable.

        :param observations: (np.array) Array of observations (eigenvalues) eigenvalues to fit kernel to.
        :param kde_bwidth: (float) The bandwidth of the kernel. (0.01 by default)
        :param kde_kernel: (str) Kernel to use [``gaussian`` by default, ``tophat``, ``epanechnikov``, ``exponential``,
                                 ``linear``,``cosine``].
        :param eval_points: (np.array) Array of values on which the fit of the KDE will be evaluated.
                                       If None, the unique values of observations are used. (None by default)
        :return: (pd.Series) Series with estimated pdf values in the eval_points.
        """

        # Reshaping array to a vertical one
        observations = observations.reshape(-1, 1)

        # Estimating Kernel Density of the empirical distribution of eigenvalues
        kde = KernelDensity(kernel=kde_kernel, bandwidth=kde_bwidth).fit(observations)

        # If no specific values provided, the fit KDE will be valued on unique eigenvalues.
        if eval_points is None:
            eval_points = np.unique(observations).reshape(-1, 1)

        # If the input vector is one-dimensional, reshaping to a vertical one
        if len(eval_points.shape) == 1:
            eval_points = eval_points.reshape(-1, 1)

        # Evaluating the log density model on the given values
        log_prob = kde.score_samples(eval_points)

        # Preparing the output of pdf values
        pdf = pd.Series(np.exp(log_prob), index=eval_points.flatten())

        return pdf

    @staticmethod
    def _get_pca(hermit_matrix):
        """
        Calculates eigenvalues and eigenvectors from a Hermitian matrix. In our case, from the correlation matrix.

        Function used to calculate the eigenvalues and eigenvectors is linalg.eigh from numpy package.

        Eigenvalues in the output are placed on the main diagonal of a matrix.

        :param hermit_matrix: (np.array) Hermitian matrix.
        :return: (np.array, np.array) Eigenvalues matrix, eigenvectors array.
        """

        # Calculating eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(hermit_matrix)

        # Index to sort eigenvalues in descending order
        indices = eigenvalues.argsort()[::-1]

        # Sorting
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        # Outputting eigenvalues on the main diagonal of a matrix
        eigenvalues = np.diagflat(eigenvalues)

        return eigenvalues, eigenvectors

    @staticmethod
    def _mp_pdf(var, tn_relation, num_points):
        """
        Derives the pdf of the Marcenko-Pastur distribution.

        Outputs the pdf for num_points between the minimum and maximum expected eigenvalues.
        Requires the variance of the distribution (var) and the relation of T - the number
        of observations of each X variable to N - the number of X variables (T/N).

        :param var: (float) Variance of the M-P distribution.
        :param tn_relation: (float) Relation of sample length T to the number of variables N (T/N).
        :param num_points: (int) Number of points to estimate pdf.
        :return: (pd.Series) Series of M-P pdf values.
        """

        # Changing the type as scipy.optimize.minimize outputs np.array with one element to this function
        if not isinstance(var, float):
            var = float(var)

        # Minimum and maximum expected eigenvalues
        eigen_min = var * (1 - (1 / tn_relation) ** (1 / 2)) ** 2
        eigen_max = var * (1 + (1 / tn_relation) ** (1 / 2)) ** 2

        # Space of eigenvalues
        eigen_space = np.linspace(eigen_min, eigen_max, num_points)

        # Marcenko-Pastur probability density function for eigen_space
        pdf = tn_relation * ((eigen_max - eigen_space) * (eigen_space - eigen_min)) ** (1 / 2) / \
              (2 * np.pi * var * eigen_space)
        pdf = pd.Series(pdf, index=eigen_space)

        return pdf

    @staticmethod
    def corr_to_cov(corr, std):
        """
        Recovers the covariance matrix from a correlation matrix.

        Requires a vector of standard deviations of variables - square root
        of elements on the main diagonal fo the covariance matrix.

        Formula used: Cov = Corr * OuterProduct(std, std)

        :param corr: (np.array) Correlation matrix.
        :param std: (np.array) Vector of standard deviations.
        :return: (np.array) Covariance matrix.
        """

        cov = corr * np.outer(std, std)
        return cov

    @staticmethod
    def cov_to_corr(cov):
        """
        Derives the correlation matrix from a covariance matrix.

        Formula used: Corr = Cov / OuterProduct(std, std)

        :param cov: (np.array) Covariance matrix.
        :return: (np.array) Covariance matrix.
        """

        # Calculating standard deviations of the elements
        std = np.sqrt(np.diag(cov))

        # Transforming to correlation matrix
        corr = cov / np.outer(std, std)

        # Making sure correlation coefficients are in (-1, 1) range
        corr[corr < -1], corr[corr > 1] = -1, 1

        return corr
