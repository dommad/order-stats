from typing import Generator
import numpy as np
import pandas as pd
from . import parsers
from .initializer import *
from .models import *
from .utils import ParserError
from .fdr import FDRCalculator, BenjaminiHochberg
from . import cythonized as cyt




class Validation:
    """Validation"""


    @staticmethod
    def get_ground_truth_label(file_name, file_tags):
        file_type = next((key for key in file_tags.keys() if key in file_name.lower()), 'random')
        return file_tags[file_type]

    @staticmethod
    def add_ground_truth_label(df, file_tag):
        df['gt_label'] = file_tag * np.ones(len(df))
        return df


    def parse_all_files(self, input_files, engine):

        ground_truth_tags = {'pos': 1, 'decoy': 2, 'negative': 0}

        try:
            parser_instance = getattr(parsers, f"{engine}Parser")()
        except AttributeError as exc:
            raise ValueError(f"Unsupported or invalid engine: {engine}") from exc


        all_dfs = []

        for file in input_files:

            file_tag = self.get_ground_truth_label(file, ground_truth_tags)

            try:
                psms_df = parser_instance.parse(file)
            except ParserError as err:
                print(f"Error parsing {file} with {engine} parser: {err}")
                continue

            psms_df = self.add_ground_truth_label(psms_df, file_tag)
            all_dfs.append(psms_df)
        
        master_df = pd.concat(all_dfs, axis=0, ignore_index=True)
        return master_df



 
    def execute_validation(self, input_file_paths: list, score: str, bootstrap_reps=500, ext_params: dict = None, engine: str = 'Tide', plot=False):
        """read the pepxml, automatically add the p-values based on lower order estimates"""

        if ext_params:
            self.params_est = ext_params


        #PARSING FILES

        # TODO: support all search engines and format, not just Comet and Tide, eliminate "mode"
        # this parsing function must be able to process target-only, randomized, decoy-only files 
        # and produce tev scores, charges, lower_order_pvalues, coute (Sidak) p_values, ground_truth_labels, i.e., labels for target, decoy, random
        #tev_scores, charges, lower_order_pvalues, coute_pvs, labels = self.__parse_get_pvals(input_files, self.params_est, option=mode)
        master_df = self.parse_all_files(input_file_paths, engine)
        # master_df.columns = ['tev', 'charge', 'hit_rank', 'lower_order_pv', 'coute(sidak)_pv', 'label', ...]

        decoy_initializer = DecoyModelInitializer(master_df[master_df['gt_label'] == 2], score)
        decoy_initializer.initialize()
        master_df = DecoyModel().calculate_p_value(master_df, score, decoy_initializer.param_dict)

        # repeat for lower-order model and CDD model, eventually abstract out to separate class
        
        # alternatively, pi_0 can be estimated as outlined in Jiang and Doerge (2008):
        # pi_0 = PiZeroEstimator().find_optimal_pi0(low_pvs, 10)
        bootstrap_results=Bootstrap(BootstrapInitializer, BenjaminiHochberg).run_bootstrap(master_df, 200)
        tprs, fdps = ProcessBootstrapResults.extract_fdps_tprs(bootstrap_results)

        bootstrap_stats = ConfidenceInterval().calculate_all_confidence_intervals(fdps, tprs, 0.32, 100)

        return bootstrap_stats


class Bootstrap:
    
    def __init__(self, initializer: Initializer, fdr_calculator: FDRCalculator) -> None:
        self.initializer = initializer
        self.fdr_calculator = fdr_calculator


    def run_bootstrap(self, df, n_rep):
        df_sorted, critical_array, pos_label, neg_label = self.initializer.initialize(df, n_rep)
        return (self.fdr_calculator.calculate_fdp_tpr(next(df_sorted), critical_array, pos_label, neg_label) for _ in range(n_rep))


class ProcessBootstrapResults:

    @staticmethod
    def extract_fdps_tprs(results: Generator):

        fdps = []
        tprs = []

        while True:
            try:
                res = next(results)
                fdps.append(res[0])
                tprs.append(res[1])
            except StopIteration:
                break

        fdps_array = np.array(fdps)
        tprs_array = np.array(tprs)

        return tprs_array, fdps_array
        

class ConfidenceInterval:

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_confidence_interval(data, idx, alpha):
        """obtain CIs from empirical bootstrap method"""

        mean = np.mean(data[:,idx])
        diff = sorted([el - mean for el in data[:,idx]])
        ci_u = mean - diff[int(len(diff) * alpha/2)]
        ci_l = mean - diff[int(len(diff) * (1- alpha/2))]

        return mean, ci_l, ci_u

    @staticmethod
    def get_confidence_interval_cython(data, idx, alpha):
        """obtain CIs from empirical bootstrap method"""
        return cyt.get_confidence_intervals(data, idx, alpha)
    

    def calculate_all_confidence_intervals(self, fdps, tprs, alpha, num_fdr_thresholds):
        fdp_cis = list(self.get_confidence_interval_cython(fdps, x, alpha) for x in range(num_fdr_thresholds))
        tpr_cis = list(self.get_confidence_interval_cython(tprs, x, alpha) for x in range(num_fdr_thresholds))

        return fdp_cis, tpr_cis


