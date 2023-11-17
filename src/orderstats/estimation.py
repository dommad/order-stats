"""Estimation of distribution parameters based on lower-order TEV distributions"""
import random
import pandas as pd
import numpy as np
from .utils import *
from . import parsers
from typing import List
from dataclasses import dataclass
from .estimators import ParametersEstimator
from .optimization_modes import ParameterOptimizationMode, LinearRegressionMode, MeanBetaMode
from .exporter import Exporter
from .cutoff_finder import CutoffFinder, MainDipCutoff
from .estimators import MethodOfMomentsEstimator, AsymptoticGumbelMLE


@dataclass
class ParametersData:

    output_dict: dict
    parameter_estimators: List
    available_charges: set



class DataFrameProcessor:

    def __init__(self) -> None:
        pass

    def get_available_hits(self, df):
        available_hits = set(df['hit_rank'])
        available_hits.discard(1) # skip first hit since it's a mixture
        return available_hits
    

    def extract_selected_hit(self, df, hit):

        df_hit = df[df['hit_rank'] == hit]
        #hit_scores = df_hit[df_hit[filter_score] > 0.01][filter_score]
        #hit_df = df_hit[df_hit[filter_score] > 0.01][filter_score] # dommad
        return df_hit
    
 
    def get_available_charges(self, df):
        return set(df['charge'])
    

    def get_psms_by_charge(self, df, charge):
        psms_by_charge = df[df['charge'] == charge].copy()
        return psms_by_charge
    
    def get_scores_below_threshold(self, df, filter_score_name, score_threshold):
        return df[df[filter_score_name] < score_threshold][filter_score_name].values


class OptimalModelsFinder:

    def __init__(self, parameters_data: ParametersData, filter_score: str) -> None:
        self.filter_score = filter_score
        self.p_data = parameters_data


    def find_parameters_for_best_estimation_optimization(self, df, df_processor: DataFrameProcessor, cutoff_finder: CutoffFinder, finding_modes: List[ParameterOptimizationMode]):
        """
        Selecting best combination of parameter estimator + finding method for each charge state
        """
        top_hits = df_processor.extract_selected_hit(df, hit=1)
        
        optimal_results = {}
    
        for charge in self.p_data.available_charges:
            charge_dict = {}
            charge_hits = df_processor.get_psms_by_charge(top_hits, charge)
            cutoff = cutoff_finder(charge_hits, self.filter_score).find_cutoff() # TODO: opt value 0.21 
            scores_below_cutoff = df_processor.get_scores_below_threshold(charge_hits, self.filter_score, cutoff)

            for p_estimator in self.p_data.parameter_estimators:
                p_estimator_name = str(p_estimator)
                param_df = self.p_data.output_dict[charge][p_estimator_name]
                
                for f_mode in finding_modes:
                    f_mode_name = str(f_mode)
                    current_opt_params = f_mode(param_df).find_optimal_parameters(scores_below_cutoff, order=0) # TODO: make order flexible
                    charge_dict[(p_estimator_name, f_mode_name)] = current_opt_params
            
            optimal_results[charge] = charge_dict

        return optimal_results


    def get_charge_best_combination_dict(self, optimal_results):
        
        best_parameters = {}
        for charge in self.p_data.available_charges:
            this_charge_results = optimal_results[charge]
            best_key = min(this_charge_results, key=lambda k: this_charge_results[k][1])
            best_parameters[charge] = (best_key, optimal_results[charge][best_key][0])

        return best_parameters




class ParametersProcessing:
    
    def __init__(self, df, df_processor: DataFrameProcessor, filter_score: str) -> None:
        
        self.df = df
        self.df_processor = df_processor
        self.filter_score = filter_score


    def process_parameters_into_charge_dicts(self, parameter_estimators=List[ParametersEstimator]):
        """calculate MLE and MM parameters using the data"""

        available_charges = self.df_processor.get_available_charges(self.df)
        output_dict = {}

        for charge in available_charges:
            charge_dict = {}
            df_by_charge = self.df_processor.get_psms_by_charge(self.df, charge)
            
            for p_estimator in parameter_estimators:
                method_params = self.find_parameters_for_lower_hits(df_by_charge, p_estimator)
                charge_dict[str(p_estimator)] = method_params

            output_dict[charge] = charge_dict

        return ParametersData(
            output_dict=output_dict,
            parameter_estimators=[str(p) for p in parameter_estimators],
            available_charges=available_charges
            )


    def find_parameters_for_lower_hits(self, df_by_charge, parameter_estimator: ParametersEstimator):
        """Get parameters either from method of moments or from MLE"""
        
        available_hits = self.df_processor.get_available_hits(df_by_charge)
        parameters = {}

        for hit in available_hits:
            hit_scores = self.df_processor.extract_selected_hit(df_by_charge, hit)[self.filter_score]
            hit_parameters = parameter_estimator(hit_scores, hit).estimate()
            parameters[hit] = hit_parameters
    
        return pd.DataFrame.from_records(parameters, index=['location', 'scale'])




class LowerOrderEstimation:
    
    def __init__(self, outname):
        self.out = outname




    # def get_bic_for_lower_models(self, lower_order_df, parameters_dict, charge, plot=False):
    #     """Calculate BIC for the lower-order models"""
    #     hit_ranks = set(lower_order_df['hit_rank'])
    #     hit_ranks.discard(1)

    #     bic_diffs = []
    #     charge_df = lower_order_df[lower_order_df['charge'] == charge].copy()
    #     charge_parameters = parameters_dict[charge]

    #     for hit_rank in hit_ranks:
    #         cur_tevs = charge_df[charge_df['hit_rank'] == hit_rank]['tev']
    #         cur_bic_diff = self.calculate_mle_mm_bic_diff(cur_tevs, hit_rank, charge_parameters)
    #         # cur_bic = self.compare_density_auc(self.tevs[idx][:,hit], mle_par, hit, idx)
    #         bic_diffs.append(cur_bic_diff)

    #         if plot:
    #             Plotting(self.out).plot_bic_diffs(bic_diffs, charge)


    # def calculate_mle_mm_bic_diff(self, tevs, hit_rank, parameters_dict, k=2):
    #     """
    #     Calculate difference between BIC for MLE and MM models
    #     for hit_rank=1
    #     """

    #     def get_bic(tevs, hit_rank, params):
    #         log_likelihood = of.AsymptoticGumbelMLE(tevs, hit_rank).get_log_likelihood(np.log(params))
    #         bic = k * np.log(len(tevs)) - 2 * log_likelihood
    #         return bic
        
    #     tevs = tevs[tevs < self.bic_cutoff]
    #     mle_params = parameters_dict['mle'][0].loc[:, hit_rank]
    #     mm_params = parameters_dict['mm'][0].loc[:, hit_rank]

    #     mle_bic = get_bic(tevs, hit_rank, mle_params)
    #     mm_bic = get_bic(tevs, hit_rank, mm_params)
    #     bic_diff = abs(mle_bic - mm_bic) / mle_bic

    #     return bic_diff

    
    # def compare_density_auc(self, tevs, parameters_dict, hit_rank):
    #     """Compare densities of the observed TEV distribution and the model"""
        
    #     mu, beta = parameters_dict['mle'][0].loc[:, hit_rank]
    #     xs, kde_observed = FFTKDE(bw=0.0005, kernel='gaussian').fit(tevs).evaluate(2**8)
    #     auc_observed = auc(xs, kde_observed)
    #     density_model = of.TEVDistribution().pdf(xs, mu, beta, hit_rank)
    #     auc_model = auc(xs, density_model)

    #     return abs(auc_model - auc_observed) / auc_observed






class PiZeroEstimator:
    """estimate pi0 for given set of p-values"""

    def __init__(self):
        pass

    @staticmethod
    def get_pi0_b(pvs, b_val):
        """calculate pi0 estimate for given b value"""
        i = 1
        condition = False

        while condition is False:
            t_i = (i-1)/b_val
            t_iplus = i/b_val
            ns_i = len(pvs[(pvs < t_iplus) & (pvs >= t_i)])
            nb_i = len(pvs[pvs >= t_i])
            condition = bool(ns_i <= nb_i/(b_val - i + 1))
            i += 1

        i -= 1

        summand = 0
        for j in range(i-1, b_val+1):
            t_j = (j-1)/b_val
            summand += len(pvs[pvs >= t_j])/((1-t_j)*len(pvs))

        pi_0 = 1/(b_val - i + 2)*summand

        return pi_0

    def get_all_pi0s(self, pvs):
        """calculate pi0 for each b value"""
        # B is from I = {5, 10, 20, 50, 100}

        pi0s = []
        b_set = [5, 10, 20, 50, 100]

        for b_val in b_set:
            pi0s.append(self.get_pi0_b(pvs, b_val))
        
        return pi0s


    def get_boostrap_pi0s(self, pvs, no_reps, b_val):

        pi0_estimates = np.zeros(no_reps)

        for rep in range(no_reps):
            random.seed()
            new_pvs = np.array(random.choices(pvs, k=len(pvs)))
            pi0_estimates[rep] = self.get_pi0_b(new_pvs, b_val)

        return pi0_estimates

    @staticmethod
    def get_mse(pi0_bootstrap, pi0_true):
        """Calculates MSE for given set of p-values and true pi0 value"""
        summand = 0
        for i in range(len(pi0_bootstrap)):
            summand += pow(pi0_bootstrap[i] - pi0_true, 2)
        
        return summand/len(pi0_bootstrap)


    def find_optimal_pi0(self, pvs, n_reps):
        """Find the optimal pi0 according to Jiang and Doerge (2008)"""
        # compute pi0 for each B
        pi0_estimates = self.get_all_pi0s(pvs)
        pi0_ave = np.mean(pi0_estimates)
        b_set = [5, 10, 20, 50, 100]

        # compute MSE for each pi0 estimate
        mses = []

        for pi0_estim, b_val in zip(pi0_estimates, b_set):
            bootstraps = self.get_boostrap_pi0s(pvs, n_reps, b_val)
            mses.append(self.get_mse(bootstraps, pi0_ave))

        optimal_idx = mses.index(sorted(mses)[0])
        print(mses)
        print(pi0_estimates)
        print(optimal_idx)
        return pi0_estimates[optimal_idx]
