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


class MainDipCutoff(CutoffFinder):

    def __init__(self, df: pd.DataFrame, filter_score: str) -> None:
        super().__init__(df, filter_score)

    def find_cutoff(self) -> float:
        """
        Find the main dip in the mixture distribution separating two components
        """
        scores = self.df[self.filter_score].values

        if len(scores) == 0:
            raise ValueError("Empty scores array. Unable to find cutoff.")

        axes, kde = FFTKDE(bw=0.05, kernel='gaussian').fit(scores).evaluate(2**8)
        dips = self.find_peaks_and_dips(axes, kde)

        if len(dips) == 0:
            main_dip = np.median(scores)
        else:
            main_dip = dips[max(0, int(len(dips / 2)) - 1)]
        
        return main_dip


    @staticmethod
    def find_peaks_and_dips(axes, kde):
        """
        Find peaks in TEV data.

        Parameters:
        - data (array-like): TEV data.

        Returns:
        - indices (array): Indices of peaks in the data.
        """

        # peaks = []
        dips = []
        for i in range(1, len(kde) - 1):
            # if kde[i] > kde[i - 1] and kde[i] > kde[i + 1]:
            #     peaks.append(i)
            if kde[i] < kde[i - 1] and kde[i] < kde[i + 1]:
                dips.append(i)

        if len(dips) == 0:
            return []
        else:
            return axes[dips] #, axes[dips]


class FixedCutoff(CutoffFinder):

    def __init__(self, df, filter_score) -> None:
        super().__init__(df, filter_score)

    def find_cutoff(self):
        return 0.21
