"""Methods to find cutoff threshold for fitting a model
to the lower-scoring portion of the mixture distribution of top-scoring PSMs"""
from abc import ABC, abstractmethod
from KDEpy import FFTKDE
import pandas as pd
import numpy as np


class CutoffFinder(ABC):

    def __init__(self, df, filter_score) -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected 'df' to be a pandas DataFrame.")
        if filter_score not in df.columns:
            raise ValueError(f"Column '{filter_score}' not found in the DataFrame.")

        self.filter_score = filter_score
        self.df: pd.DataFrame = df

    @abstractmethod
    def find_cutoff(self):
        pass



class PeakDipFinder:

    @staticmethod
    def find_peaks_and_dips(axes, kde):
        """
        Find peaks in TEV data.

        Parameters:
        - data (array-like): TEV data.

        Returns:
        - indices (array): Indices of peaks in the data.
        """

        # Compute differences between consecutive elements
        diff = np.diff(kde)

        # Find indices of peaks and dips
        # peaks = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0] + 1
        dips = np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0] + 1
        
        # dips = np.where((kde[1:-1] < kde[:-2]) & (kde[1:-1] < kde[2:]))[0] + 1
        return axes[dips] # , axes[peaks]



class MainDipCutoff(CutoffFinder, PeakDipFinder):
    """Finding the cutoff by idenifying the main dip between negative and positive
    components of the mixture distribution"""
    # TODO: fix it, it doesn't work properly now

    def find_cutoff(self) -> float:
        """
        Find the main dip in the mixture distribution separating two components
        """
        scores = self.df[self.filter_score].values

        if len(scores) == 0:
            raise ValueError("Empty scores array. Unable to find cutoff.")

        axes, kde = FFTKDE(bw=0.05, kernel='gaussian').fit(scores).evaluate(2**8)
        dips = self.find_peaks_and_dips(axes, kde)

        return np.median(scores) if dips == [] else dips[max(0, int(len(dips / 2)) - 1)]



class FixedCutoff(CutoffFinder):

    def find_cutoff(self):
        return 0.18