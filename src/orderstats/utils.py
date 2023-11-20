import time
import numpy as np
from pandas import DataFrame, Series
from abc import ABC, abstractmethod


TH_N0 = 1000.
TH_MU = 0.02 * np.log(TH_N0)
TH_BETA = 0.02

NUM_CHARGES_TO_EXPORT = 10


def fetch_instance(class_name, attribute_name, *args, **kwargs):
    """general fetches for class attributes by name and possibly initializing them"""
    try:
        return getattr(class_name, attribute_name)(*args, **kwargs)
    except AttributeError as exc:
        raise ValueError(f"Unsupported or invalid instance type: {class_name}") from exc


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds.")
        return result
    return wrapper


def log_function_call(func):

    def wrapper(*args, **kwargs):
        #print(f"Calling {func.__name__} with args {args} and kwargs {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result

    return wrapper


def calculate_tev(df: DataFrame, par_a: float, par_n0: float) -> Series:
    """
    Calculate the log-transformed e-value (TEV) score based on the given parameters.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing relevant information.
    - par_a (float): The 'a' parameter used in TEV score calculation.
    - par_n0 (float): The 'N0' parameter used in TEV score calculation.

    Returns:
    np.ndarray: An array containing TEV scores for each row in the DataFrame.
    """

    if 'e_value' in df.columns:
        return par_a * np.log(df['e_value'] / par_n0)

    return par_a * np.log(df['p_value'] * df['num_candidates'] / par_n0)


def _is_numeric(value):
    if not isinstance(value, str):
        return False
    try:
        float(value)
        return True

    except ValueError:
        return False


def largest_factors(n):
    for i in range(n // 2, 0, -1):
        if n % i == 0:
            return n // i, i


class StrClassNameMeta(type):

    def __str__(cls):
        return cls.__name__
    

class PValueCalculator(ABC):

    @abstractmethod
    def calculate_p_value(self, df_to_modify, score_column, param_dict):
        pass


class SidakCorrectionMixin:

    @staticmethod
    def sidak_correction(df, p_val_column):
        df[f"sidak_{p_val_column}"] = 1 - pow(1 - df[p_val_column].values, df["num_candidates"].values)
        return df


class ParserError(Exception):
    """A custom exception class."""
    def __init__(self, message="An error occurred."):
        self.message = message
        super().__init__(self.message)

