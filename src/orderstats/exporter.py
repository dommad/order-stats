from abc import ABC, abstractmethod
import pandas as pd
from .utils import *


class Exporter(ABC):

    @abstractmethod
    def export_parameters(self):
        pass


class PeptideProphetExporter(ABC):

    def __init__(self, out_name) -> None:
        self.out_name = out_name

    def export_parameters(self, params_est):

        """export params to txt for modified PeptideProphet (mean & std)"""
        try:
            params = pd.DataFrame.from_dict(params_est, orient='index', columns=['location', 'scale'])
            params['location'] += params['scale'] * np.euler_gamma # convert to mean
            params['scale'] = np.pi / np.sqrt(6) * params['scale'] # conver to std
            params.loc[1, :] = params.iloc[0, :] # add parameters for charge 1+ that we didn't consider

            # add parameters from highest present charge state to all missing higher charge states
            max_idx = max(params.index)
            len_missing = NUM_CHARGES_TO_EXPORT - max_idx
            new_idx = range(max_idx + 1, NUM_CHARGES_TO_EXPORT + 1)

            to_concat = pd.DataFrame(len_missing * (params.loc[max_idx, :],), index = new_idx, columns=['location', 'scale'])
            final_params = pd.concat([params, to_concat], axis=0, ignore_index=False)
            final_params.sort_index(inplace=True)

            final_params.to_csv(f"pp_params_{self.out}.txt", sep=" ", header=None, index=None)
        
        except Exception as e:
            print(f"Error occurred during parameter export: {str(e)}")


