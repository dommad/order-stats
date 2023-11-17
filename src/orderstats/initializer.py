from abc import ABC, abstractmethod
from .estimation import OptimalModelsFinder, ParametersProcessing
from . import parsers


class Initializer:
    pass

class EstimationInitializer(Initializer):

    def initialize_parser(self, engine):
        return getattr(parsers, f"{engine}Parser")()

    def initialize_parameter_estimators(self, *args):
        return args

    def initialize_param_processing(self, df, df_processor, filter_score):
        return ParametersProcessing(df, df_processor, filter_score)

    def initialize_optimal_models_finder(self, param_dict, filter_score):
        return OptimalModelsFinder(param_dict, filter_score)

    def initialize_optimization_modes(self, *args):
        return args