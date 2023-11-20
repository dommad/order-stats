"""Estimation of distribution parameters based on lower-order TEV distributions"""
from typing import List
from dataclasses import dataclass
import pandas as pd
from .utils import *
from .estimators import ParametersEstimator
from .optimization_modes import ParameterOptimizationMode
from .cutoff_finder import CutoffFinder
# from .exporter import Exporter


@dataclass
class ParametersData:

    output_dict: dict
    parameter_estimators: List
    available_charges: set



class Processor:
    pass

class DataFrameProcessor(Processor):

    def __init__(self) -> None:
        pass

    def get_available_hits(self, df):
        all_hit_ranks = df.drop_duplicates('hit_rank')['hit_rank'].values
        all_hit_ranks = all_hit_ranks[all_hit_ranks != 1] # skip top hit since it may contain some correct IDs
        return all_hit_ranks
    

    def extract_selected_hit(self, df, hit):

        df_hit = df[df['hit_rank'] == hit]
        #hit_scores = df_hit[df_hit[filter_score] > 0.01][filter_score]
        #hit_df = df_hit[df_hit[filter_score] > 0.01][filter_score] # dommad
        return df_hit
    
    def extract_hit_scores(self, df, score, hit):
        return df.loc[df['hit_rank'].values == hit, score].values
    
 
    def get_available_charges(self, df):
        return df.drop_duplicates('charge')['charge'].values
    

    def get_psms_by_charge(self, df, charge):
        return df[df['charge'].values == charge]
    
    def get_scores_below_threshold(self, df, score_name, threshold):
        return df.loc[df[score_name].values < threshold, score_name].values


class OptimalModelsFinder:

    def __init__(self, config, parameters_data: ParametersData, filter_score: str) -> None:
        self.filter_score = filter_score
        self.p_data = parameters_data
        self.config = config


    def find_parameters_for_best_estimation_optimization(self, df, df_processor: DataFrameProcessor, optimization_modes: List[ParameterOptimizationMode]):
        """
        Selecting best combination of parameter estimator + finding method for each charge state
        """
        cutoff_generator = fetch_instance(CutoffFinder, self.config.get('estimation', 'cutoff_finder').strip(), None)
        top_hits = df_processor.extract_selected_hit(df, hit = 1)
        
        optimal_results = {}
    
        for charge in self.p_data.available_charges:
            charge_dict = {}
            charge_hits = df_processor.get_psms_by_charge(top_hits, charge)
            cutoff = cutoff_generator(charge_hits, self.filter_score).find_cutoff() # TODO: opt value 0.21 
            scores_below_cutoff = df_processor.get_scores_below_threshold(charge_hits, self.config.get('general', 'filter_score', fallback='tev'), cutoff)

            for p_estimator in self.p_data.parameter_estimators:
                p_estimator_name = p_estimator.__name__ # it was str(p_estimator)
                param_df = self.p_data.output_dict[charge][p_estimator_name]
                
                for mode in optimization_modes:
                    opt_mode_name = mode.__name__ # it was str(mode)
                    current_opt_params = mode(param_df).find_optimal_parameters(scores_below_cutoff, hit_rank=1) # TODO: make order flexible
                    charge_dict[(p_estimator_name, opt_mode_name)] = current_opt_params
            
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
    
    def __init__(self, config, df, df_processor: DataFrameProcessor, filter_score: str) -> None:
        
        self.df = df
        self.df_processor = df_processor
        self.filter_score = filter_score
        self.config = config


    def process_parameters_into_charge_dicts(self, parameter_estimators: List[ParametersEstimator]):
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
            parameter_estimators=[p.__name__ for p in parameter_estimators], # it was str(p) for p...
            available_charges=available_charges
            )


    def find_parameters_for_lower_hits(self, df_by_charge, parameter_estimator: ParametersEstimator):
        """Get parameters either from method of moments or from MLE"""
        
        available_hits = self.df_processor.get_available_hits(df_by_charge)
        parameters = {}
    
        for hit in available_hits:
            hit_scores = self.df_processor.extract_hit_scores(df_by_charge, self.filter_score, hit)
            hit_parameters = parameter_estimator(hit_scores, hit).estimate()
            parameters[hit] = hit_parameters
    
        return pd.DataFrame.from_records(parameters, index=['location', 'scale'])


# class LowerOrderEstimation:
    
#     def __init__(self, outname):
#         self.out = outname




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







