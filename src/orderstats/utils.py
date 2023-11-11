import random
from collections import deque
import functools as fu
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
from xml.etree import ElementTree as ET
from pyteomics import pepxml, mzid
from KDEpy import FFTKDE
from sklearn.metrics import auc



TH_N0 = 1000.
TH_MU = 0.02 * np.log(TH_N0)
TH_BETA = 0.02