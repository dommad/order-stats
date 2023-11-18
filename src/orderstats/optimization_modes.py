from abc import ABC, ABCMeta, abstractmethod
import numpy as np
from scipy import optimize as opt, stats as st
from pandas import DataFrame
from .utils import StrClassNameMeta
from . import stat


class ParameterOptimizationModeMeta(StrClassNameMeta, ABCMeta):
    pass


class ParameterOptimizationMode(ABC, metaclass=ParameterOptimizationModeMeta):

    @abstractmethod
    def generate_optimization_helper(self):
        pass

    @abstractmethod
    def find_optimal_parameters(self):
        pass



class LinearRegressionMode(ParameterOptimizationMode):

    def __init__(self, parameters_df: DataFrame = None, lr_cutoff=()) -> None:
        super().__init__()
        # TODO: lr_cutoff may be obsolete if we implement the best_linreg mode
        self.param_df = parameters_df
        self.lr_cutoff = lr_cutoff


    def generate_optimization_helper(self):
        """Fit linear regression to mu and beta values 
        from lower orders and return the results"""

        if self.param_df is None or not isinstance(self.param_df, DataFrame):
            raise ValueError("Invalid or missing parameter DataFrame.")

        columns_within_cutoff = self.param_df.columns.isin(range(*self.lr_cutoff))
        selected_df = self.param_df.loc[:, columns_within_cutoff]
        mu_values = selected_df.loc['location', :]
        beta_values = selected_df.loc['scale', :]
        linreg_results = st.linregress(mu_values, beta_values)

        return linreg_results
    

    def find_optimal_parameters(self, data, hit_rank):

        if not isinstance(data, np.ndarray) or not isinstance(hit_rank, int):
            raise ValueError("Invalid input data or hit_rank.")

        linreg = self.find_best_linear_regression()

        def get_log_likelihood(params, data, hit_rank):
            """
            Optimizing BIC for the lower section of the distribution
            is equivalent to optimizing log-likelihood
            """
            mu = max(params[0], 1e-6)  # Extract mu from the parameters
           
            beta = mu * linreg.slope + linreg.intercept
            beta = max(beta, 0.001)

            logged_params = np.log(np.array([mu, beta]))

            log_like = stat.AsymptoticGumbelMLE(data, hit_rank).get_log_likelihood(logged_params)
            return log_like

        
        initial_guess = np.mean(data)
        bounds = [(0.1 * initial_guess, 1.5 * initial_guess)]

        objective_f = lambda params, scores, hit_rank: get_log_likelihood(params, scores, hit_rank)

        results = opt.minimize(
            fun = objective_f,
            x0 = np.array([initial_guess]),
            args=(data, hit_rank,),
            method='L-BFGS-B',
            bounds=bounds   
            )
        
        opt_mu = results.x[0]
        opt_beta = opt_mu * linreg.slope + linreg.intercept
        if opt_beta < 0:
            return (opt_mu, opt_beta), 10 # is beta negative, it's useless
     
        return (opt_mu, opt_beta), results.fun
    

    def find_best_linear_regression(self):
        """Fit linear regression to mu and beta values 
        from lower orders and return the results for the best lr_cutoff"""

        if self.param_df is None or not isinstance(self.param_df, DataFrame):
            raise ValueError("Invalid or missing parameter DataFrame.")

        x, y, n_size = 3, len(self.param_df.columns), 5  # Adjust the values as needed
        range_values = range(x, y)
        lr_cutoff_slices = [range_values[i:i+n_size] for i in range(len(range_values) - n_size + 1)]


        best_linreg_results = None
        best_lr_cutoff = None
        max_pearson_r = -1  # Initialize with a value less than any possible Pearson's R

        # Iterate over lr_cutoff values
        for lr_cutoff in lr_cutoff_slices:

            selected_df = self.param_df.loc[:, self.param_df.columns.isin(lr_cutoff)]
            mu_values = selected_df.loc['location', :].values
            beta_values = selected_df.loc['scale', :].values

            # Fit linear regression
            linreg_results = st.linregress(mu_values, beta_values)

            # Check if the current linear regression has a higher Pearson's R
            if abs(linreg_results.rvalue) > max_pearson_r:
                max_pearson_r = abs(linreg_results.rvalue)
                best_linreg_results = linreg_results
                best_lr_cutoff = lr_cutoff

        return best_linreg_results



class MeanBetaMode(ParameterOptimizationMode):

    def __init__(self, parameters_df: DataFrame = None, min_rank=8) -> None:
        super().__init__()
        self.min_rank = min_rank
        self.param_df = parameters_df


    def generate_optimization_helper(self):
        """get linear regression parameters and mean beta for further processing"""

        if self.param_df is None or not isinstance(self.param_df, DataFrame):
            raise ValueError("Invalid or missing parameter DataFrame.")
        
        columns_within_cutoff = self.param_df.columns >= self.min_rank
        mean_beta = self.param_df.loc['scale', columns_within_cutoff].mean()

        return mean_beta


    def find_optimal_parameters(self, scores, hit_rank):
        """
        Maximum Likelihood Estimation for TEV distribution
        """
        if not isinstance(scores, np.ndarray) or not isinstance(hit_rank, int):
            raise ValueError("Invalid input data or hit_rank.")

        beta = self.generate_optimization_helper()

        def get_log_likelihood(params, data, hit_rank):
            """
            Optimizing BIC for the lower section of the distribution
            is equivalent to optimizing log-likelihood
            """
            mu = max(params[0], 1e-6)  # Extract mu from the parameters
            logged_params = np.log(np.array([mu, beta]))
            log_like = stat.AsymptoticGumbelMLE(data, hit_rank).get_log_likelihood(logged_params)
            return log_like

        
        initial_guess = np.mean(scores)
        bounds = [(0.1 * initial_guess, 1.5 * initial_guess)]

        objective_f = lambda params, scores, hit_rank: get_log_likelihood(params, scores, hit_rank)

        results = opt.minimize(
            fun = objective_f,
            x0 = np.array([initial_guess]),
            args=(scores, hit_rank,),
            method='L-BFGS-B',
            bounds=bounds   
            )
        
        opt_mu = results.x[0]
        opt_beta = beta
        
        return (opt_mu, opt_beta), results.fun
    
