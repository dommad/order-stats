"""Full analysis of pepxml file using lower order statistics"""
from .utils import *
from . import parsers
from .initializer import Initializer
from .estimators import *
from .optimization_modes import *


def run_estimation(engine, input_file, cutoff_filter, df_processor, init: Initializer):
    # parsing
    try:
        parser_instance = getattr(parsers, f"{engine}Parser")()
    except AttributeError:
        raise ValueError(f"Unsupported or invalid engine: {engine}")

    filter_score = 'tev'

    parser_instance = init.initialize_parser(engine)
    df = parser_instance.parse(input_file)
    df['tev'] = calculate_tev(df, -TH_BETA, TH_N0, engine)

    parameter_estimators = init.initialize_parameter_estimators(AsymptoticGumbelMLE, MethodOfMomentsEstimator)
    para_init = init.initialize_param_processing(df, df_processor, filter_score)

    param_data = para_init.process_parameters_into_charge_dicts(parameter_estimators)

    optimal_finder = init.initialize_optimal_models_finder(param_data, filter_score)
    sel_find_modes = init.initialize_optimization_modes(LinearRegressionMode, MeanBetaMode)

    optimal_results = optimal_finder.find_parameters_for_best_estimation_optimization(df, df_processor, cutoff_filter, sel_find_modes)
    best_params = optimal_finder.get_charge_best_combination_dict(optimal_results)

    return best_params