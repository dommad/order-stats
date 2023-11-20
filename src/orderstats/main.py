"""Full analysis of pepxml file using lower order statistics"""
import argparse
import configparser
from .utils import TH_BETA, TH_N0, calculate_tev
from .initializer import Initializer, EstimationInitializer
from .estimation import Processor
from .exporter import Exporter



def fetch_instance(class_name, attribute_name, *args, **kwargs):
    """general fetches for class attributes by name and possibly initializing them"""
    try:
        return getattr(class_name, attribute_name)(*args, **kwargs)
    except AttributeError as exc:
        raise ValueError(f"Unsupported or invalid instance type: {class_name}") from exc
    

def run_estimation(config_file_path,
                   input_file,
                   df_processor: Processor,
                   init: Initializer):
    """The main function running estimation of distribution parameters
    for top-scoring target PSMs using lower-order statistics"""

    with open(config_file_path, 'r', encoding='utf-8') as config_file:
        config = configparser.ConfigParser()
        config.read_file(config_file)

    df_processor = fetch_instance(Processor, df_processor, None)
    init = EstimationInitializer(config, df_processor)

    parser = init.initialize_parser()
    df = parser.parse(input_file)
    df['tev'] = calculate_tev(df, -TH_BETA, TH_N0)


    parameter_estimators = init.initialize_parameter_estimators()
    para_init = init.initialize_param_processing(df)
    param_data = para_init.process_parameters_into_charge_dicts(parameter_estimators)

    opt_finder = init.initialize_optimal_models_finder(param_data)
    opt_modes = init.initialize_optimization_modes()

    optimal_results = opt_finder.find_parameters_for_best_estimation_optimization(df, df_processor, opt_modes)
    best_params = opt_finder.get_charge_best_combination_dict(optimal_results)

    exporter = fetch_instance(Exporter, f"{config.get('estimation', 'exporter', fallback='PeptideProphet')}Exporter", None)
    exporter(config.get('general', 'outname', fallback="example")).export_parameters(best_params)

    return best_params, param_data, df



if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description='Lower-Order Statistics for FDR Estimation in Shotgun Proteomics')
    arg_parser.add_argument('-conf', '--configuration_file', required=True, type=str,
                            help="Configuration file in TOML format")
    arg_parser.add_argument('-p',  '--positives_file', required=True,
                        type=str, help='file(s) with positive ground truth samples (accepted format: pep.xml, tsv, txt, mzid)')
    arg_parser.add_argument('-n', '--negatives_file', required=True,
                        type=str, help="file(s) with negative ground truth samples (accepted format: pep.xml, tsv, txt, mzid")
    arg_parser.add_argument('-d',  '--decoys_file', required=True,
                        type=str, help='file(s) with results of decoy-only search (accepted format: pep.xml, tsv, txt, mzid)')

    args = arg_parser.parse_args()