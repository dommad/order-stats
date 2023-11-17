"""Distribution functions, parameter estimation codes"""
import math
from functools import reduce
from scipy import special, stats as st, optimize as opt
import numpy as np
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
from .utils import *

class TEVDistribution:
    """Methods related to Gumbel distribution"""

    @staticmethod
    def pdf(x_vals, location, scale, order_index):
        """
        PDF for TEV distribution of the specified order
        """

        scale = max(scale, 1e-6)
        z_vals = (x_vals - location) / scale
        denominator = scale * math.factorial(order_index)
        pdf = (1 / denominator) * np.exp(-(order_index + 1) * z_vals - np.exp(-z_vals))

        return pdf


    @staticmethod
    def gumbel_new_ppf(p_value, location, scale):
        """quantile function for Gumbel distribution"""
        return location - scale * np.log(-np.log(p_value))


    @staticmethod
    def cdf_asymptotic(x_val, location, scale, k):
        """
        Cumulative Distribution Function (CDF) of the Truncated Exponential Value (TEV) distribution of order k
        for the mu-beta parametrization.

        Parameters:
        - x_val (float or array-like): Values at which to evaluate the CDF.
        - location (float): Location parameter of the TEV distribution.
        - scale (float): Scale parameter of the TEV distribution.
        - k (int): Order of the TEV distribution.

        Returns:
        - cdf (float or array-like): Calculated CDF values at the given x_val.

        The function calculates the CDF of the TEV distribution of order k using the mu-beta parametrization. It employs
        an asymptotic expansion approach for efficiency.

        Example usage:
        ```python
        x_values = np.linspace(0, 10, 100)
        location_param = 2.0
        scale_param = 1.5
        order = 2
        tev_cdf = cdf_asymptotic(x_values, location_param, scale_param, order)
        ```
        """

        z_val = (np.array(x_val) - location) / scale
        factorials = map(lambda m: np.exp(-m * z_val) / math.factorial(m), range(k + 1))
        summed = reduce(lambda x, y: x + y, factorials)
        cdf = np.exp(-np.exp(-z_val)) * summed

        return cdf


    @staticmethod
    def cdf_finite_n(x_val, mu_, beta, k, n_cand):
        """CDF of TEV distribution of order k for finite N form"""
        z_val = (np.array(x_val)-mu_)/beta
        fx_ = 1 - (1/n_cand)*np.exp(-z_val)
        summands = np.zeros(len(z_val))
        for idx in range(k+1):
            cur_sum = special.comb(n_cand, n_cand-idx)*pow(fx_, n_cand-idx)*pow(1-fx_, idx)
            summands += cur_sum
        return summands



class MLE:
    """Maximum Likelihood Estimation for TEV distributions"""

    def __init__(self, data, order):
        self.data = data
        self.order = order
        self.initial_guess = np.array([np.mean(data), np.std(data)])

    def get_log_likelihood(self):
        pass

    def run_mle(self):
        """
        Maximum Likelihood Estimation for TEV distribution
        """

        results = opt.minimize(
            fun = self.get_log_likelihood,
            x0 = self.initial_guess,
            method = 'Nelder-Mead'
            )

        location, scale = np.exp(results.x)
        return location, scale
  


class MethodOfMoments:

    def __init__(self) -> None:
        pass


    def estimate_parameters(self, data, k):
        """
        Method of Moments (MM) estimates of location and scale parameters for the specified order.

        Parameters:
        - data (array-like): Input data.
        - order (int): Order parameter.

        Returns:
        - location (float): Estimated location parameter.
        - scale (float): Estimated scale parameter.
        """

        euler_m = -special.digamma(1)
        first_moment = self._get_first_moment(data)
        second_moment = self._get_second_moment(data)
        scale  = np.sqrt((second_moment - pow(first_moment, 2) )/self._get_trigamma(k))
        location = first_moment - scale * (euler_m - self._get_sum_inverse_squared(k))

        return location, scale


    def _get_trigamma(self, k):
        """
        Calculate the trigamma function.

        Parameters:
        - k (int): Input parameter.

        Returns:
        - trigamma_value (float): Trigamma function value.
        """
        return math.pi**2 / 6 - self._get_sum_inverse_squared(k + 1)


    @staticmethod
    def _get_first_moment(data):
        """
        Calculate the raw first moment.

        Parameters:
        - data (array-like): Input data.

        Returns:
        - first_moment (float): Raw first moment.
        """

        return np.mean(data)


    @staticmethod
    def _get_second_moment(data):
        """
        Calculate the raw second moment.

        Parameters:
        - data (array-like): Input data.

        Returns:
        - second_moment (float): Raw second moment.
        """

        squared_values = np.square(data)
        length = max(1, len(data))
        return (1 / length) * np.sum(squared_values)


    @staticmethod
    def _get_sum_inverse_squared(k):
        """
        Calculate the sum of squared values.

        Parameters:
        - k (int): Input parameter.

        Returns:
        - sum_squared_value (float): Sum of squared values.
        """
        squared_values = 1 / np.square(np.arange(1, k + 1))
        return np.sum(squared_values)



class AsymptoticGumbelMLE(MLE):
    
    def __init__(self, data, order):
        super().__init__(data, order)
    

    def get_log_likelihood(self, log_params):
        """
        Log-likelihood function for TEV distribution
        """

        num_points = len(self.data)
        location, scale = np.exp(log_params) + 1e-10
        z_values = (self.data - location) / scale
        factorial_term = math.factorial(self.order)
        likelihood = -num_points * np.log(scale * factorial_term) - (self.order + 1) * np.sum(z_values) - np.sum(np.exp(-z_values))
        return -likelihood


class FiniteNGumbelMLE(MLE):
    
    def __init__(self, data, order, num_candidates):
        super().__init__(data, order)
        self.num_candidates = num_candidates


    def get_log_likelihood(self, log_params):
        """
        Log-likelihood function for the TEV distribution with finite N form,
        where N is the same for all data points.

        Parameters:
        - log_params (array-like): Log-transformed parameters [log(location), log(scale)].

        Returns:
        - log_likelihood (float): Log-likelihood value.
        """

        num_data_points = len(self.data)
        location, scale = np.exp(log_params) + 1e-10
        z_values = (self.data - location) / scale

        cdf_term = 1 - (1 / self.num_candidates) * np.exp(-z_values)
        pdf_term = 1 / (self.num_candidates * scale) * np.exp(-z_values)
        remaining_term = self.num_candidates - self.order

        term1 = num_data_points * np.log(remaining_term * special.comb(self.num_candidates, remaining_term))
        term2 = np.sum(np.log(pdf_term)) + (remaining_term - 1) * np.sum(np.log(cdf_term))
        term3 = (self.num_candidates - remaining_term) * np.sum(np.log(1 - cdf_term))

        log_likelihood = term1 + term2 + term3

        return -log_likelihood



class EMAlgorithm:
    """
    Expectation-Maximization (EM) algorithm for estimating parameters of a mixture distribution.

    This class implements the EM algorithm to estimate the parameters of a mixture distribution composed of a Gumbel
    (negative) component and a Gaussian (positive) component. The algorithm iteratively maximizes the likelihood function
    by updating the parameters in the Expectation (E) step and the Maximization (M) step.

    Attributes:
    - None

    Methods:
    - initialize_parameters(data): Initialize parameters for the EM algorithm based on the input data.
    - run_em_algorithm(data, fixed_gumbel=False, fixed_gumbel_params=None, max_iterations=300, tolerance=1e-6): Execute the EM
      algorithm to estimate parameters of the mixture distribution.
    - find_peaks_and_dips(axes, kde): Find peaks and dips in TEV data.
    - estimate_gumbel_params(data, weights=None): Estimate parameters for the Gumbel distribution.
    - estimate_gaussian_params(data, weights=None): Estimate parameters for the Gaussian distribution.
    - get_gumbel_neg_log_likelihood(params, data, weights): Calculate the negative log-likelihood for the Gumbel distribution.
    - get_gaussian_neg_log_likelihood(params, data, weights): Calculate the negative log-likelihood for the Gaussian
      distribution.

    Example usage:
    ```python
    em_algo = EMAlgorithm()
    initial_params = em_algo.initialize_parameters(data)
    mu1, beta1, mu2, sigma2, pi0 = em_algo.em_algorithm(data)
    ```
    """

    def __init__(self, data):
        self.data = data


    def initialize_parameters(self):
        """
        Initialize parameters for the EM algorithm.

        Parameters:
        - data (array-like): Input data.

        Returns:
        - mu1 (float): Initial location parameter for Gumbel component.
        - beta1 (float): Initial scale parameter for Gumbel component.
        - mu2 (float): Initial mean parameter for Gaussian component.
        - sigma2 (float): Initial standard deviation for Gaussian component.
        - pi0 (float): Initial proportion of data from Gumbel component.
        """

        # Find the peaks and dips
        axes, kde = FFTKDE(bw=0.05, kernel='gaussian').fit(self.data).evaluate(2**8)
        peaks, dips = self.find_peaks_and_dips(axes, kde)

        if len(dips) == 0:
            main_dip = np.median(self.data)
        else:
            main_dip = dips[max(0, int(len(dips / 2)) - 1)]

        gumbel_data = self.data[self.data < main_dip]
        gaussian_data = self.data[self.data >= main_dip]

        # Initial parameter estimates
        mu1 = peaks[0]
        beta1 = np.std(gumbel_data)
        mu2 = peaks[-1]
        sigma2 = np.std(gaussian_data)
        pi0 = len(gumbel_data) / len(self.data)

        return mu1, beta1, mu2, sigma2, pi0


    def run_em_algorithm(self, fixed_gumbel=False, fixed_gumbel_params=None, max_iterations=30, tolerance=1e-6):
        """
        Execute the EM algorithm to estimate parameters of the mixture distribution.

        Parameters:
        - data (array-like): Input data.
        - fixed_gumbel (bool): If True, fix Gumbel component parameters.
        - fixed_gumbel_params (tuple): Tuple of fixed Gumbel parameters if fixed_gumbel is True.
        - max_iterations (int): Maximum number of iterations for EM algorithm.
        - tolerance (float): Convergence tolerance.

        Returns:
        - mu1 (float): Estimated location parameter for Gumbel.
        - beta1 (float): Estimated scale parameter for Gumbel.
        - mu2 (float): Estimated mean parameter for Gaussian.
        - sigma2 (float): Estimated standard deviation for Gaussian.
        - pi0 (float): Estimated proportion of data from Gumbel component.
        """

        mu1, beta1, mu2, sigma2, pi0 = self.initialize_parameters()

        if fixed_gumbel:
            mu1, beta1 = fixed_gumbel_params

        for iteration in range(max_iterations):
            # E-step
            pi1 = 1 - pi0
            pdf_gumbel = st.gumbel_r.pdf(self.data, mu1, beta1) * pi0
            pdf_gaussian = st.norm.pdf(self.data, mu2, sigma2) * pi1

            # Calculate responsibilities
            gamma = pdf_gumbel / (pdf_gumbel + pdf_gaussian)

            # M-step
            # Update parameters
            if not fixed_gumbel:
                mu1, beta1 = self.estimate_gumbel_params(gamma)

            mu2, sigma2 = self.estimate_gaussian_params(1 - gamma)
            pi0 = np.mean(gamma)

            # Check for convergence
            if np.abs(pi0 - gamma).max() < tolerance:
                break

        return mu1, beta1, mu2, sigma2, pi0


    @staticmethod
    def find_peaks_and_dips(axes, kde):
        """
        Find peaks in TEV data.

        Parameters:
        - data (array-like): TEV data.

        Returns:
        - indices (array): Indices of peaks in the data.
        """

        peaks = []
        dips = []
        for i in range(1, len(kde) - 1):
            if kde[i] > kde[i - 1] and kde[i] > kde[i + 1]:
                peaks.append(i)
            if kde[i] < kde[i - 1] and kde[i] < kde[i + 1]:
                dips.append(i)

        return axes[peaks], axes[dips]


    def estimate_gumbel_params(self, weights=None):
        """
        Estimate Gumbel distribution parameters.

        Parameters:
        - data (array-like): Input data.
        - weights (array-like): Weights for each data point.

        Returns:
        - mu (float): Estimated location parameter for Gumbel.
        - beta (float): Estimated scale parameter for Gumbel.
        """

        if weights is None:
            weights = np.ones_like(self.data)

        mu_init, beta_init = np.median(self.data), np.std(self.data) / np.sqrt(6)
        params = opt.minimize(self.get_gumbel_neg_log_likelihood, (mu_init, beta_init), args=(weights),
                          method='Nelder-Mead')
        mu, beta = params.x

        return mu, beta



    def estimate_gaussian_params(self, weights=None):
        """
        Estimate Gaussian distribution parameters.

        Parameters:
        - data (array-like): Input data.
        - weights (array-like): Weights for each data point.

        Returns:
        - mu (float): Estimated mean parameter for Gaussian.
        - sigma (float): Estimated standard deviation for Gaussian.
        """

        if weights is None:
            weights = np.ones_like(self.data)

        mu_init, sigma_init = np.mean(self.data), np.std(self.data)
        params = opt.minimize(self.get_gaussian_neg_log_likelihood, (mu_init, sigma_init), args=(weights),
                          method='Nelder-Mead')
        mu, sigma = params.x

        return mu, sigma


    def get_gumbel_neg_log_likelihood(self, params, weights):
        """
        Negative log-likelihood for Gumbel distribution.

        Parameters:
        - params (tuple): Gumbel parameters (mu, beta).
        - data (array-like): Input data.
        - weights (array-like): Weights for each data point.

        Returns:
        - neg_log_likelihood (float): Negative log-likelihood.
        """

        mu, beta = params
        gumbel_pdf = st.gumbel_r.pdf(self.data, mu, beta)
        neg_log_likelihood = -np.sum(weights * np.log(gumbel_pdf))

        return neg_log_likelihood


    def get_gaussian_neg_log_likelihood(self, params, weights):
        """
        Negative log-likelihood for Gaussian distribution.

        Parameters:
        - params (tuple): Gaussian parameters (mu, sigma).
        - data (array-like): Input data.
        - weights (array-like): Weights for each data point.

        Returns:
        - neg_log_likelihood (float): Negative log-likelihood.
        """

        mu, sigma = params
        gaussian_pdf = st.norm.pdf(self.data, mu, sigma)
        neg_log_likelihood = -np.sum(weights * np.log(gaussian_pdf))

        return neg_log_likelihood


    @staticmethod
    def plot_em_results(em_results, data, out_name):
        """
        Plot the results of the Expectation-Maximization (EM) algorithm for a mixture distribution.

        Parameters:
        - em_results (tuple): Tuple containing the estimated parameters (mu1, beta1, mu2, sigma2, pi0) from the EM algorithm.
        - data (array-like): Input data.
        - out_name (str): The name to be used when saving the plot.

        Returns:
        - None: The function saves the plot as a PNG file.

        The function visualizes the EM algorithm results by plotting the observed data distribution alongside the estimated
        components of a mixture distribution. It displays the positive component (modeled as a Gaussian distribution),
        the negative component (modeled as a Gumbel distribution), and the observed data distribution.

        Parameters:
        - em_results (tuple): A tuple containing the following parameters:
            - mu1 (float): Location parameter for the negative (Gumbel) component.
            - beta1 (float): Scale parameter for the negative (Gumbel) component.
            - mu2 (float): Mean parameter for the positive (Gaussian) component.
            - sigma2 (float): Standard deviation for the positive (Gaussian) component.
            - pi0 (float): Proportion of data from the negative (Gumbel) component.

        - data (array-like): Input data used in the EM algorithm.

        - out_name (str): The base name (without extension) to be used when saving the plot. The function will save the plot as
        a PNG file with the name '{out_name}.png'.

        Note: The function uses the FFTKDE method to estimate the kernel density of the observed data.

        """
    
        def get_fftkde(data, pi0):
            axes, kde = FFTKDE(bw=0.0005, kernel='gaussian').fit(data).evaluate(2**8)
            return axes, pi0 * kde

        domain = np.linspace(0, 1, 300)
        mu1, beta1, mu2, sigma2, pi0 = em_results

        plt.style.use('tableau-colorblind10')
        plt.rcParams.update({'font.size': 12, 'font.family': 'Helvetica'})

        fig, ax = plt.subplots(figsize=(6, 5))
        
        ax.plot(domain, (1 - pi0) * st.norm.pdf(domain, mu2, sigma2), label='positive')
        ax.plot(domain, pi0 * st.gumbel_r.pdf(domain, mu1, beta1), label='negative')
        ax.plot(*get_fftkde(data, pi0), label='observed')
        
        ax.set_xlim(0, 0.8)
        ax.set_xlabel("TEV")
        ax.set_ylabel("density")
        ax.legend()

        fig.savefig(f"{out_name}.png", dpi=500, bbox_inches="tight")
