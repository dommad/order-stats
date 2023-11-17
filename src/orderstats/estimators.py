from abc import ABC, ABCMeta, abstractmethod
from . import stat
from .utils import StrClassNameMeta


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

