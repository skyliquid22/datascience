# pylint: disable=protected-access
"""
Tests the functions from the FilterMatrix class.
"""

import warnings
import unittest
import os
import numpy as np
import pandas as pd
from datascience.filter.filter import FilterMatrix


class TestFilterMatrix(unittest.TestCase):
    """
    Tests different functions of the Filter Matrix class.
    """

    def setUp(self):
        """
        Initialize and get the test data
        """

        # Stock prices data to test the Covariance functions
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/dataset/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

        # Now we can calculate the returns from the prices
        self.returns = self.data.pct_change()
        self.returns.dropna(inplace=True, how='all')

    def test_mp_pdf(self):
        """
        Test the deriving of pdf of the Marcenko-Pastur distribution.
        """

        filt = FilterMatrix()

        # Properties for the distribution
        var = 0.1
        tn_relation = 5
        num_points = 5

        # Calculating the pdf in 5 points
        pdf_mp = filt._mp_pdf(var, tn_relation, num_points)

        # Testing the minimum and maximum and non-zero values of the pdf
        self.assertAlmostEqual(pdf_mp.index[0], 0.03056, delta=1e-4)
        self.assertAlmostEqual(pdf_mp.index[4], 0.20944, delta=1e-4)

        # Testing that the distribution curve is right
        self.assertTrue(pdf_mp.values[1] > pdf_mp.values[2] > pdf_mp.values[3])

    def test_fit_kde(self):
        """
        Test the kernel fitting to a series of observations.
        """

        filt = FilterMatrix()

        # Values to fit kernel to and evaluation points
        observations = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5])
        eval_points = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        # Calculating the pdf in 7 chosen points
        pdf_kde = filt._fit_kde(observations, eval_points=eval_points, kde_bwidth=0.25)

        # Testing the values and if the pdf is symmetric
        self.assertEqual(pdf_kde[0.0], pdf_kde[0.6])
        self.assertEqual(pdf_kde[0.1], pdf_kde[0.5])
        self.assertEqual(pdf_kde[0.2], pdf_kde[0.4])
        self.assertAlmostEqual(pdf_kde[0.3], 1.44413, delta=1e-5)

        # Testing also on unique values of the set as a default output
        pdf_kde_default = filt._fit_kde(observations, kde_bwidth=0.25)
        self.assertEqual(pdf_kde[0.1], pdf_kde_default[0.1])
        self.assertEqual(pdf_kde_default[0.2], pdf_kde_default[0.4])

    def test_pdf_fit(self):
        """
        Test the fit between empirical pdf and the theoretical pdf.
        """

        filt = FilterMatrix()

        # Values to calculate theoretical and empirical pdfs
        var = 0.6
        eigen_observations = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5])
        tn_relation = 2
        kde_bwidth = 0.4

        # Calculating the SSE
        pdf_kde = filt._pdf_fit(var, eigen_observations, tn_relation, kde_bwidth)

        # Testing the SSE value
        self.assertAlmostEqual(pdf_kde, 50.51326, delta=1e-5)

    def test_find_max_eval(self):
        """
        Test the search for maximum random eigenvalue.
        """

        filt = FilterMatrix()

        # Values to calculate theoretical and empirical pdfs
        eigen_observations = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5])
        tn_relation = 2
        kde_bwidth = 0.4

        # Optimizing and getting the maximum random eigenvalue and the optimal variation
        maximum_eigen, var = filt._find_max_eval(eigen_observations, tn_relation, kde_bwidth)

        # Testing the maximum random eigenvalue and the optimal variation
        self.assertAlmostEqual(maximum_eigen, 2.41011, delta=1e-5)
        self.assertAlmostEqual(var, 0.82702, delta=1e-5)

    @staticmethod
    def test_corr_to_cov():
        """
        Test the recovering of the covariance matrix from the correlation matrix.
        """

        filt = FilterMatrix()

        # Correlation matrix and the vector of standard deviations
        corr_matrix = np.array([[1, 0.1, -0.1],
                                [0.1, 1, -0.3],
                                [-0.1, -0.3, 1]])
        std_vec = np.array([0.1, 0.2, 0.1])

        # Expected covariance matrix
        expected_matrix = np.array([[0.01, 0.002, -0.001],
                                    [0.002, 0.04, -0.006],
                                    [-0.001, -0.006, 0.01]])

        # Finding the covariance matrix
        cov_matrix = filt.corr_to_cov(corr_matrix, std_vec)

        # Testing the first row of the matrix
        np.testing.assert_almost_equal(cov_matrix, expected_matrix, decimal=5)

    @staticmethod
    def test_cov_to_corr():
        """
        Test the deriving of the correlation matrix from a covariance matrix.
        """

        filt = FilterMatrix()

        # Covariance matrix
        cov_matrix = np.array([[0.01, 0.002, -0.001],
                               [0.002, 0.04, -0.006],
                               [-0.001, -0.006, 0.01]])

        # Expected correlation matrix
        expected_matrix = np.array([[1, 0.1, -0.1],
                                    [0.1, 1, -0.3],
                                    [-0.1, -0.3, 1]])

        # Finding the covariance matrix
        corr_matrix = filt.cov_to_corr(cov_matrix)

        # Testing the first row of the matrix
        np.testing.assert_almost_equal(corr_matrix, expected_matrix, decimal=5)

    @staticmethod
    def test_get_pca():
        """
        Test the calculation of eigenvalues and eigenvectors from a Hermitian matrix.
        """

        filt = FilterMatrix()

        # Correlation matrix as an input
        corr_matrix = np.array([[1, 0.1, -0.1],
                                [0.1, 1, -0.3],
                                [-0.1, -0.3, 1]])

        # Expected correlation matrix
        expected_eigenvalues = np.array([[1.3562, 0, 0],
                                         [0, 0.9438, 0],
                                         [0, 0, 0.7]])
        first_eigenvector = np.array([-3.69048184e-01, -9.29410263e-01, 1.10397126e-16])

        # Finding the eigenvalues
        eigenvalues, eigenvectors = filt._get_pca(corr_matrix)

        # Testing eigenvalues and the first eigenvector
        np.testing.assert_almost_equal(eigenvalues, expected_eigenvalues, decimal=4)
        np.testing.assert_almost_equal(eigenvectors[0], first_eigenvector, decimal=5)

    @staticmethod
    def test_denoised_corr_spectral():
        """
        Test the shrinkage the eigenvalues associated with noise.
        """

        filt = FilterMatrix()

        # Eigenvalues and eigenvectors to use
        eigenvalues = np.array([[1.3562, 0, 0],
                                [0, 0.9438, 0],
                                [0, 0, 0.7]])
        eigenvectors = np.array([[-3.69048184e-01, -9.29410263e-01, 1.10397126e-16],
                                 [-6.57192300e-01, 2.60956474e-01, 7.07106781e-01],
                                 [6.57192300e-01, -2.60956474e-01, 7.07106781e-01]])

        # Expected correlation matrix
        expected_corr = np.array([[1, 1, -1],
                                  [1, 1, -1],
                                  [-1, -1, 1]])

        # Finding the de-noised correlation matrix
        corr_matrix = filt._denoised_corr_spectral(eigenvalues, eigenvectors, 1)

        # Testing if the de-noised correlation matrix is right
        np.testing.assert_almost_equal(corr_matrix, expected_corr, decimal=4)

    def test_filter_corr_hierarchical(self):
        """
        Test the filtering of the emperical correlation matrix.
        """

        filt = FilterMatrix()

        # Correlation matrix to test
        corr = np.array([[1, 0.70573243, 0.03085437, 0.6019651, 0.81214341],
                         [0.70573243, 1, 0.03126594, 0.56559443, 0.88961155],
                         [0.03085437, 0.03126594, 1, 0.01760481, 0.02842086],
                         [0.60196510, 0.56559443, 0.01760481, 1, 0.73827921],
                         [0.81214341, 0.88961155, 0.02842086, 0.73827921, 1]])

        expected_corr_avg = np.array([[1, 0.44618396, 0.44618396, 0.44618396, 0.61711376],
                                      [0.44618396, 1, 0.29843018, 0.29843018, 0.61711376],
                                      [0.44618396, 0.29843018, 1, 0.01760481, 0.61711376],
                                      [0.44618396, 0.29843018, 0.01760481, 1, 0.61711376],
                                      [0.61711376, 0.61711376, 0.61711376, 0.61711376, 1]])

        expected_corr_single = np.array([[1, 0.03126594, 0.03085437, 0.03085437, 0.03085437],
                                         [0.03126594, 1, 0.03126594, 0.03126594, 0.03126594],
                                         [0.03085437, 0.03126594, 1, 0.01760481, 0.02842086],
                                         [0.03085437, 0.03126594, 0.01760481, 1, 0.02842086],
                                         [0.03085437, 0.03126594, 0.02842086, 0.02842086, 1]])

        expected_corr_complete = np.array([[1, 0.70573243, 0.70573243, 0.70573243, 0.88961155],
                                           [0.70573243, 1, 0.56559443, 0.56559443, 0.88961155],
                                           [0.70573243, 0.56559443, 1, 0.01760481, 0.88961155],
                                           [0.70573243, 0.56559443, 0.01760481, 1, 0.88961155],
                                           [0.88961155, 0.88961155, 0.88961155, 0.88961155, 1]])

        methods_list = ['complete', 'single', 'average']
        # Compute all methods with given correlation matrix
        corr_complete, corr_single, corr_average = [filt.filter_corr_hierarchical(corr, methods) for methods in methods_list]

        # Test plot
        filt.filter_corr_hierarchical(corr, draw_plot=True)

        # Testing is filtered matrices are consistent with expected values.
        np.testing.assert_almost_equal(corr_complete, expected_corr_complete, decimal=4)
        np.testing.assert_almost_equal(corr_single, expected_corr_single, decimal=4)
        np.testing.assert_almost_equal(corr_average, expected_corr_avg, decimal=4)

        # Testing input matrix with invalid inputs.
        bad_dimension = np.array([1, 0])
        bad_size = np.array([[1, 0, 1], [0, 1, 1]])
        non_positive = np.array([[1, -1], [0, 1]])
        non_sym = np.array([[0, 0], [0, 0]])

        bad_inputs = [bad_dimension, bad_size, non_positive, non_sym]

        # Testing for warnings
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            bad_inputs_results = [filt.filter_corr_hierarchical(bads) for bads in bad_inputs]
            self.assertEqual(len(warn), 4)

        bad_inputs.append(corr)

        # Testing with invalid method parameter
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            bad_inputs_results.append(filt.filter_corr_hierarchical(corr, method='bad'))
            self.assertEqual(len(warn), 1)

        # Testing to see if failed return fetches the unfiltered correlation array
        for idx, result in enumerate(bad_inputs_results):
            np.testing.assert_almost_equal(result, bad_inputs[idx], decimal=4)

    @staticmethod
    def test_denoise_covariance():
        """
        TODO
        Test the shrinkage the eigenvalues associated with noise.
        """
