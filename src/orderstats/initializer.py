from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import scipy.stats as st
import numpy as np
from .estimation import OptimalModelsFinder, ParametersProcessing
from . import estimators, optimization_modes, parsers, exporter
from .constants import GROUND_TRUTH_TAGS
from .utils import fetch_instance
from configparser import ConfigParser


class Initializer:

    @abstractmethod
    def initialize(self):
        pass
    

class EstimationInitializer:

    def __init__(self, config: ConfigParser, df_processor) -> None:
        self.config = config
        self.engine = config.get('general', 'engine', fallback='Tide')
        self.filter_score = config.get('general', 'filter_score', fallback='tev')
        self.df_processor = df_processor


    def initialize_parser(self):
        try:
            return getattr(parsers, f"{self.engine}Parser")()
        except AttributeError as exc:
            raise ValueError(f"Unsupported or invalid engine: {self.engine}") from exc

        #return getattr(parsers, f"{self.engine}Parser")()


    def initialize_parameter_estimators(self):
        estimator_names = self.config.get('estimation', 'estimators').strip().split(', ')
        # TODO: add error handling
        estimator_objects = [getattr(estimators, x) for x in estimator_names]
        return estimator_objects


    def initialize_param_processing(self, df):
        return ParametersProcessing(self.config, df, self.df_processor, self.filter_score)


    def initialize_optimal_models_finder(self, param_dict):
        return OptimalModelsFinder(self.config, param_dict, self.filter_score)


    def initialize_optimization_modes(self):
        opt_mode_names = self.config.get('estimation', 'optimization_modes').strip().split(', ')
        # TODO: add error handling
        optimization_objects = [getattr(optimization_modes, f"{x}Mode") for x in opt_mode_names]
        return optimization_objects
    


class ValidationInitializer:

    def __init__(self, config: ConfigParser, df_processor) -> None:
        self.config = config
        self.engine = config.get('general', 'engine', fallback='Tide')
        self.filter_score = config.get('general', 'filter_score', fallback='tev')
        self.df_processor = df_processor



class ExporterInitializer:

    def __init__(self, config) -> None:
        self.exporter_name = f"{config.get('estimation', 'exporter', fallback='PeptideProphet').strip()}Exporter"
        self.out_name = config.get('general', 'outname', fallback='test').strip()

    def initialize(self):
        try:
            return fetch_instance(exporter, self.exporter_name, self.out_name)
        except AttributeError as exc:
            raise ValueError("Unsupported or invalid exporter.") from exc


class ModelInitializer(ABC):

    @abstractmethod
    def initialize(self):
        pass


class CDDModelInitializer(ModelInitializer):

    def __init__(self, parameters: Union[str, pd.DataFrame]) -> None:
        self.param_input = parameters
        self.param_dict = {}

    def initialize(self):
        if isinstance(self.param_input, str):
            self.param_dict = parsers.ParamFileParser(self.param_input).parse()
        elif isinstance(self.param_input, pd.DataFrame):
            self.param_dict = self.param_input
        else:
            raise TypeError("Parameters input format is unsupported.")
        

class DecoyModelInitializer(ModelInitializer):

    def __init__(self, decoy_df: pd.DataFrame, score_column):
        self.decoy_df = decoy_df
        self.score_column = score_column
        self.param_dict = {}

    def initialize(self):
        if not isinstance(self.decoy_df, pd.DataFrame):
            raise TypeError("The input provided is not a pandas DataFrame.")
            
        self.param_dict = self.decoy_df[self.decoy_df['hit_rank'] == 1].groupby('charge')[self.score_column].apply(st.gumbel_r.fit).to_dict()


class LowerOrderModelInitializer(ModelInitializer):

    def __init__(self, parameters: Union[str, pd.DataFrame]) -> None:
        self.param_input = parameters
        self.param_dict = {}

    def initialize(self):
        if isinstance(self.param_input, str):
            self.param_dict = parsers.ParamFileParser(self.param_input).parse()
        # it's possible for the user to provide parameters directly as pandas dataframe
        elif isinstance(self.param_input, pd.DataFrame):
            self.param_dict = self.param_input
        else:
            raise TypeError("Parameters input format is unsupported.")
        

class BootstrapInitializer(Initializer):

    @staticmethod
    def initialize(df: pd.DataFrame, n_rep, fdr_threshold_array):

        length_df = len(df)

        bootstrap_idxs = [np.random.choice(np.arange(length_df), size=length_df, replace=True) for _ in range(n_rep)]
        critical_vals = np.arange(1, length_df + 1) / length_df
        critical_list = [critical_vals * x for x in fdr_threshold_array]
        bootstrapped = [df.iloc[x, :] for x in bootstrap_idxs]
        sorted_dfs = (x.iloc[np.argsort(x.p_value.values)] for x in bootstrapped)

        return sorted_dfs, critical_list, GROUND_TRUTH_TAGS['positive'], GROUND_TRUTH_TAGS['negative']
    



