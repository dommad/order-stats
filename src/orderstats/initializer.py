from abc import ABC, abstractmethod
from .estimation import OptimalModelsFinder, ParametersProcessing
from . import parsers


class Initializer:
    pass

class EstimationInitializer(Initializer):

    def __init__(self, engine, df_processor, filter_score) -> None:
        self.engine = engine
        self.df_processor = df_processor
        self.filter_score = filter_score

    def initialize_parser(self):
        try:
            parser_instance = getattr(parsers, f"{self.engine}Parser")()
        except AttributeError:
            raise ValueError(f"Unsupported or invalid engine: {self.engine}")
    
        return getattr(parsers, f"{self.engine}Parser")()

    def initialize_parameter_estimators(self, *args):
        return args

    def initialize_param_processing(self, df):
        return ParametersProcessing(df, self.df_processor, self.filter_score)

    def initialize_optimal_models_finder(self, param_dict):
        return OptimalModelsFinder(param_dict, self.filter_score)

    def initialize_optimization_modes(self, *args):
        return args