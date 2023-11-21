"""Constants to be used by estimation and validation functions"""
import numpy as np

TH_N0 = 1000.
TH_MU = 0.02 * np.log(TH_N0)
TH_BETA = 0.02

NUM_CHARGES_TO_EXPORT = 10

GROUND_TRUTH_TAGS =  {'positive': 1, 'decoy': 2, 'negative': 0, 'unidentified': 3}
