"""Exporters"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .utils import NUM_CHARGES_TO_EXPORT


class Exporter(ABC):
    """General class for exporters of parameters estimated for top null models"""

    @abstractmethod
    def export_parameters(self, params_data):
        """Method responsible for exporting the parameters estimated for top null models"""
        pass

class MuBetaExporter(Exporter):
    """Exporter for parameters of top null models in the form: charge: mu, beta"""

    def __init__(self, out_name: str):
        self.out_name = out_name
        self.params: pd.DataFrame = None


    def export_parameters(self, params_data: dict):
        """Export parameters to txt file."""

        try:
            final_params = self.create_params_dataframe(params_data)
            self.save_to_txt(final_params)

        except (ValueError, IOError, PermissionError) as e:
            print(f"Error occurred during parameter export: {str(e)}")


    def save_to_txt(self, final_params: pd.DataFrame):
        """Save parameters to txt file."""
        final_params.to_csv(f"mu_beta_params_{self.out_name}.txt", sep=" ", header=None, index=None)


    def create_params_dataframe(self, params_data: dict):
        """Create DataFrame from parameters data."""

        extracted_params = dict((key, val[1]) for key, val in params_data.items())
        params = pd.DataFrame.from_dict(extracted_params, orient='index', columns=['location', 'scale'])
        params.loc[1, :] = params.iloc[0, :] # add parameters for charge 1+ that we didn't consider
        final_params = self.fill_in_missing_charges(params)

        return final_params


    def fill_in_missing_charges(self, params_df: pd.DataFrame):
        """Fill in missing charges in the DataFrame."""

        # add parameters from highest present charge state to all missing higher charge states
        max_idx = max(params_df.index)
        len_missing = NUM_CHARGES_TO_EXPORT - max_idx
        new_idx = range(max_idx + 1, NUM_CHARGES_TO_EXPORT + 1)

        to_concat = pd.DataFrame(len_missing * (params_df.loc[max_idx, :],), index = new_idx, columns=['location', 'scale'])
        final_params = pd.concat([params_df, to_concat], axis=0, ignore_index=False)
        final_params.sort_index(inplace=True)

        return final_params


class PeptideProphetExporter(Exporter):
    """Exporter for parameters of top null models in the form: charge: mean, std"""

    def __init__(self, out_name: str):
        self.out_name = out_name
        self.params: pd.DataFrame = None


    def export_parameters(self, params_data: dict):
        """Export parameters to txt file."""

        try:
            final_params = self.create_params_dataframe(params_data)
            self.save_to_txt(final_params)

        except (ValueError, IOError, PermissionError) as e:
            logging.error(f"Error occurred during parameter export: {str(e)}")


    def save_to_txt(self, final_params: pd.DataFrame):
        """Save parameters to txt file."""
        final_params.to_csv(f"pp_params_{self.out_name}.txt", sep=" ", header=None, index=None)


    def create_params_dataframe(self, params_data: dict):
        """Create DataFrame from parameters data."""

        extracted_params = dict((key, val[1]) for key, val in params_data.items())
        params = pd.DataFrame.from_dict(extracted_params, orient='index', columns=['location', 'scale'])
        params['location'] += params['scale'] * np.euler_gamma # convert to mean
        params['scale'] = np.pi / np.sqrt(6) * params['scale'] # conver to std
        params.loc[1, :] = params.iloc[0, :] # add parameters for charge 1+ that we didn't consider
        final_params = self.fill_in_missing_charges(params)

        return final_params


    def fill_in_missing_charges(self, params_df: pd.DataFrame):
        """Fill in missing charges in the DataFrame."""

        # add parameters from highest present charge state to all missing higher charge states
        max_idx = max(params_df.index)
        len_missing = NUM_CHARGES_TO_EXPORT - max_idx
        new_idx = range(max_idx + 1, NUM_CHARGES_TO_EXPORT + 1)

        to_concat = pd.DataFrame(len_missing * (params_df.loc[max_idx, :],), index = new_idx, columns=['location', 'scale'])
        final_params = pd.concat([params_df, to_concat], axis=0, ignore_index=False)
        final_params.sort_index(inplace=True)

        return final_params
