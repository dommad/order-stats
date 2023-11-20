from abc import ABC, ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import scipy.stats as st
from .utils import StrClassNameMeta
from . import stat


class ParametersEstimatorMeta(StrClassNameMeta, ABCMeta):
    pass


class ParametersEstimator(ABC, metaclass=ParametersEstimatorMeta):

    def __init__(self, scores, hit_index) -> None:
        self.scores = scores
        self.hit_index = hit_index

    @abstractmethod
    def estimate(self):
        pass


class MethodOfMomentsEstimator(ParametersEstimator):

    def __init__(self, scores, hit_index) -> None:
        super().__init__(scores, hit_index)

    def estimate(self):
        return stat.MethodOfMoments().estimate_parameters(self.scores, self.hit_index)
    

class AsymptoticGumbelMLE(ParametersEstimator):

    def __init__(self, scores, hit_index) -> None:
        super().__init__(scores, hit_index)

    def estimate(self):
        return stat.AsymptoticGumbelMLE(self.scores, self.hit_index).run_mle()


class FiniteNGUmbelMLE(ParametersEstimator):

    def __init__(self, scores, hit_index, num_candidates=1000) -> None:
        super().__init__(scores, hit_index)
        self.num_candidates = num_candidates

    def estimate(self):
        return stat.FiniteNGumbelMLE(self.scores, self.hit_index, self.num_candidates)
    


class PiZeroEstimator(ABC):

    @abstractmethod
    def calculate_pi_zero(self):
        pass



class BootstrapPiZero(PiZeroEstimator):
    """estimate pi0 for given set of p-values"""

    def __init__(self):
        pass


    def calculate_pi_zero(self, pvs, n_reps):
        pi0_estimates = np.array(self.get_all_pi0s(pvs))
        pi0_ave = np.mean(pi0_estimates)
        b_set = [5, 10, 20, 50, 100]

        mses = [self.get_mse(self.get_bootstrap_pi0s(pvs, n_reps, b_val), pi0_ave) for b_val in b_set]

        optimal_idx = np.argmin(mses)
        return pi0_estimates[optimal_idx]

    @staticmethod
    def get_pi0_b(pvs, b_val):
        i = 1

        while True:
            t_i = (i - 1) / b_val
            t_iplus = i / b_val
            ns_i = np.sum((t_i <= pvs) & (pvs < t_iplus))
            nb_i = np.sum(pvs >= t_i)

            if ns_i <= nb_i / (b_val - i + 1):
                break

            i += 1

        i -= 1
        t_values = [(j - 1) / b_val for j in range(i, b_val + 1)]
        pi_0 = np.sum([np.sum(pvs >= t) / ((1 - t) * len(pvs)) for t in t_values]) / (b_val - i + 2)

        return pi_0

    # @staticmethod
    # def get_pi0_b(pvs, b_val):
    #     """calculate pi0 estimate for given b value"""
    #     i = 1
    #     condition = False

    #     while condition is False:
    #         t_i = (i-1)/b_val
    #         t_iplus = i/b_val
    #         ns_i = len(pvs[(pvs < t_iplus) & (pvs >= t_i)])
    #         nb_i = len(pvs[pvs >= t_i])
    #         condition = bool(ns_i <= nb_i/(b_val - i + 1))
    #         i += 1

    #     i -= 1

    #     summand = 0
    #     for j in range(i-1, b_val+1):
    #         t_j = (j-1)/b_val
    #         summand += len(pvs[pvs >= t_j])/((1-t_j)*len(pvs))

    #     pi_0 = 1/(b_val - i + 2)*summand

    #     return pi_0

    def get_all_pi0s(self, pvs):
        b_set = [5, 10, 20, 50, 100]
        return [self.get_pi0_b(pvs, b_val) for b_val in b_set]

    # def get_all_pi0s(self, pvs):
    #     """calculate pi0 for each b value"""
    #     # B is from I = {5, 10, 20, 50, 100}

    #     pi0s = []
    #     b_set = [5, 10, 20, 50, 100]

    #     for b_val in b_set:
    #         pi0s.append(self.get_pi0_b(pvs, b_val))
        
    #     return pi0s


    def get_bootstrap_pi0s(self, pvs, no_reps, b_val):
        return np.array([self.get_pi0_b(np.random.choice(pvs, size=len(pvs)), b_val) for _ in range(no_reps)])


    # def get_boostrap_pi0s(self, pvs, no_reps, b_val):

    #     pi0_estimates = np.zeros(no_reps)

    #     for rep in range(no_reps):
    #         random.seed()
    #         new_pvs = np.array(random.choices(pvs, k=len(pvs)))
    #         pi0_estimates[rep] = self.get_pi0_b(new_pvs, b_val)

    #     return pi0_estimates


    @staticmethod
    def get_mse(pi0_bootstrap, pi0_true):
        return np.mean((pi0_bootstrap - pi0_true)**2)


class TruePiZero(PiZeroEstimator):

    @staticmethod
    def calculate_pi_zero(df):
        pi_0 = len(df[np.where(df.labels == 0)])/len(df[np.where(df.labels != 2)])
        return pi_0


class CoutePiZero(PiZeroEstimator):
    
    @staticmethod
    def calculate_pi_zero(coute_pvs):
        """Get pi0 estimate using the graphical method describe in Coute et al."""
        compl_pvs = 1 - coute_pvs
        sorted_compls = np.sort(compl_pvs)
        dfs = pd.DataFrame(sorted_compls, columns=['score'])
        dfs.index += 1
        dfs['cdf'] = dfs.index/len(dfs)

        l_lim = int(0.4*len(dfs))
        u_lim = int(0.6*len(dfs))
        lr_ = st.linregress(dfs['score'][l_lim:u_lim], dfs['cdf'][l_lim:u_lim])
        return lr_.slope
    
