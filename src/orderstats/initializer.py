from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import scipy.stats as st
import numpy as np
from .estimation import OptimalModelsFinder, ParametersProcessing
from . import parsers



class Initializer:

    @abstractmethod
    def initialize(self):
        pass
    

class EstimationInitializer:

    def __init__(self, engine, df_processor, filter_score) -> None:
        self.engine = engine
        self.df_processor = df_processor
        self.filter_score = filter_score

    def initialize_parser(self):
        try:
            return getattr(parsers, f"{self.engine}Parser")()
        except AttributeError as exc:
            raise ValueError(f"Unsupported or invalid engine: {self.engine}") from exc

        #return getattr(parsers, f"{self.engine}Parser")()

    def initialize_parameter_estimators(self, *args):
        return args

    def initialize_param_processing(self, df):
        return ParametersProcessing(df, self.df_processor, self.filter_score)

    def initialize_optimal_models_finder(self, param_dict):
        return OptimalModelsFinder(param_dict, self.filter_score)

    def initialize_optimization_modes(self, *args):
        return args
    

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
            parsers.ParamFileParser(self.param_input).read_param_file()
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

    def __init__(self, parameters: Union[str, pd.DataFrame], score_column) -> None:
        self.param_input = parameters
        self.score_column = score_column
        self.param_dict = {}

    def initialize(self):
        if isinstance(self.param_input, str):
            parsers.ParamFileParser(self.param_input).read_param_file()
        elif isinstance(self.param_input, pd.DataFrame):
            self.param_dict = self.param_input
        else:
            raise TypeError("Parameters input format is unsupported.")
        

class BootstrapInitializer(Initializer):

    @staticmethod
    def initialize(df: pd.DataFrame, n_rep):

        fdr_nums = 100

        th_array = np.linspace(0.001, 0.1, fdr_nums)
        label_dict = {'positive': 0, 'negative': 1}
        length_df = len(df)

        bootstrap_idxs = [np.random.choice(np.arange(length_df), size=length_df, replace=True) for _ in range(n_rep)]
        critical_vals = np.arange(1, length_df + 1) / length_df
        critical_list = [critical_vals * x for x in th_array]
        bootstrapped = [df.iloc[x, :] for x in bootstrap_idxs]
        sorted_dfs = (x.iloc[np.argsort(x.p_value.values)] for x in bootstrapped)

        return sorted_dfs, critical_list, label_dict['positive'], label_dict['negative']
    



