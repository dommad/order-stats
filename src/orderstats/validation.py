from typing import Generator, List
import numpy as np
import pandas as pd
from . import parsers
from . import initializer
from .utils import ParserError, PValueCalculator, SidakCorrectionMixin, calculate_tev, TH_BETA, TH_N0
from .fdr import FDRCalculator, BenjaminiHochberg
from . import cythonized as cyt


CONFIDENCE_INTERVAL_ALPHA = 0.32
NUM_FDR_THRESHOLDS = 100



class Validation:
    """Validation"""


    def __init__(self, decoy_model, lower_order_model, cdd_model, cdd_params, lower_params, fdr_method: FDRCalculator, filter_score, n_rep) -> None:

        self.decoy_model = decoy_model
        self.lower_order_model = lower_order_model
        self.cdd_model = cdd_model
        self.cdd_params = cdd_params
        self.lower_params = lower_params
        self.fdr_method = fdr_method
        self.n_rep = n_rep
        self.filter_score = filter_score

    
    def fetch_instance(self, class_name, attribute_name, *args, **kwargs):
        """general fetches for class attributes by name and possibly initializing them"""
        try:
            return getattr(class_name, attribute_name)(*args, **kwargs)
        except AttributeError as exc:
            raise ValueError(f"Unsupported or invalid instance type: {class_name}") from exc

    @staticmethod
    def get_ground_truth_label(file_name, file_tags):
        """Extracting ground truth label based on the name of the file"""
        label = next((k for k in file_tags.keys() if k in file_name.lower()), 'unidentified')
        return file_tags[label]

    @staticmethod
    def add_ground_truth_label(df: pd.DataFrame, file_tag: int) -> pd.DataFrame:
        """Adding ground truth label to the dataframe"""
        df['gt_label'] = file_tag * np.ones(len(df))
        return df


    def parse_all_files(self, input_files: List[str], engine: str) -> pd.DataFrame:
        """Helper function to parse all files"""

        # TODO: this should be customizable by the user in config file
        ground_truth_tags = {'pos': 1, 'decoy': 2, 'negative': 0, 'unidentified': 3}
        parser_instance = self.fetch_instance(parsers, f"{engine}Parser", )
     
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


    def calculate_p_values(self, df_to_modify: pd.DataFrame, model: PValueCalculator, sidak: bool = False, param_df: pd.DataFrame = None, parameters_file: str = None):
        """Initialize model and use it to calculate p-values for the provided dataframe"""
        
        if param_df is not None:
            init_instance = self.fetch_instance(initializer, f"{model.__name__}Initializer", param_df, self.filter_score)
        elif parameters_file is not None:
            init_instance = self.fetch_instance(initializer, f"{model.__name__}Initializer", parameters_file)
        else:
            raise ValueError("No dataframe or parameters file provided, so cannot initialize the model for p-value calculation.")
        
        init_instance.initialize()
        df_with_pvs, pv_column_name = model.calculate_p_value(df_to_modify, self.filter_score, init_instance.param_dict)

        if issubclass(model, SidakCorrectionMixin) and sidak:
            df_with_pvs = model.sidak_correction(df_with_pvs, pv_column_name)

        return df_with_pvs
    

    def initialize_and_run_bootstrap(self, df, p_value_column):
        """Initialize the Bootstrap instance and run the bootstrap"""
        bootstrap_instance = Bootstrap(initializer.BootstrapInitializer, self.fdr_method, p_value_column)
        return bootstrap_instance.run_bootstrap(df, self.n_rep)

 
    def execute_validation(self, input_file_paths: list, engine: str = 'Tide'):
        """read the pepxml, automatically add the p-values based on lower order estimates"""

        master_df = self.parse_all_files(input_file_paths, engine)
        master_df = master_df[master_df['hit_rank'] == 1]
        master_df['tev'] = calculate_tev(master_df, -TH_BETA, TH_N0)
        master_df.loc[(master_df.gt_label == 1) & (master_df.tev < 0.3), 'gt_label'] = 2 # just for testing
        master_df = self.calculate_p_values(master_df, self.decoy_model, param_df=master_df[master_df['gt_label'] == 2], sidak=True)
        master_df = self.calculate_p_values(master_df, self.lower_order_model, sidak=True, parameters_file=self.lower_params)
        master_df = self.calculate_p_values(master_df, self.cdd_model, sidak=True, parameters_file=self.cdd_params)

        # alternatively, pi_0 can be estimated as outlined in Jiang and Doerge (2008):
        # pi_0 = PiZeroEstimator().find_optimal_pi0(low_pvs, 10)
        # add pi_0 adjustment for benjamini hochberg

        bootstrap_results = self.initialize_and_run_bootstrap(master_df, 'CDD_p_value')
        tprs, fdps = ProcessBootstrapResults.extract_fdps_tprs(bootstrap_results)

        bootstrap_stats = ConfidenceInterval().calculate_all_confidence_intervals(fdps, tprs, CONFIDENCE_INTERVAL_ALPHA, NUM_FDR_THRESHOLDS)

        return bootstrap_stats, master_df



class Bootstrap:
    """Bootstrap"""
    
    def __init__(self, init: initializer.Initializer, fdr_calculator: FDRCalculator, p_value_column: str) -> None:
        self.init = init
        self.fdr_calculator = fdr_calculator
        self.p_value_column = p_value_column


    def run_bootstrap(self, df, n_rep):
        """main method to run bootstrap"""
        df_sorted, critical_array, pos_label, neg_label = self.init.initialize(df, n_rep)
        return (self.fdr_calculator.calculate_fdp_tpr(next(df_sorted), self.p_value_column, critical_array, pos_label, neg_label) for _ in range(n_rep))



class ProcessBootstrapResults:
    """Processing Bootstrap results"""

    @staticmethod
    def extract_fdps_tprs(results: Generator):
        """extract FDP and TPR values from the bootstrapping output"""

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
    """Calculating confidence intervals on the bootstrapped FDR and FDP"""

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
        """calculate CIs for both FDP and TPR"""
        fdp_cis = list(self.get_confidence_interval_cython(fdps, x, alpha) for x in range(num_fdr_thresholds))
        tpr_cis = list(self.get_confidence_interval_cython(tprs, x, alpha) for x in range(num_fdr_thresholds))

        return fdp_cis, tpr_cis


