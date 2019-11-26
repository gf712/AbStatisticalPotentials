import itertools
from .heuristics import CombinedHeuristics

class ApproximateStatisticalPotential:
    def __init__(self, heuristics):
        self._heuristics = heuristics
        self._upper_computations, self._lower_computations, self._upper_idx, self._lower_idx = \
            self.approx_w_heuristics(heuristics)
        self._name = f"sp_{'_'.join(x.name for x in self._heuristics)}"

    def possible_element_combinations(self):
        return itertools.product(*[h.value_set for h in self._heuristics], repeat=1)

    def serialization_name(self):
        return '|'.join(x.serialization_name() for x in self._heuristics)

    @property
    def upper_computations(self):
        return self._upper_computations
        
    @property
    def lower_computations(self):
        return self._lower_computations

    @property
    def upper_idx(self):
        return self._upper_idx
    
    @property
    def lower_idx(self):
        return self._lower_idx
    
    @property
    def name(self):
        return self._name
    
    @staticmethod
    def approx_w_heuristics(h):
        if len(h) < 2:
            raise ValueError("Expected h to be a dict with two or more entries.")
        upper = list()
        lower = list()
        upper_idx = list()
        lower_idx = list()
        for x in range(1, len(h) + 1):
            if x % 2 == 1:
                upper.extend([CombinedHeuristics(combo) for combo in itertools.combinations(h, r=x)])
                [upper_idx.append(combo) for combo in itertools.combinations(range(len(h)), r=x)]
            else:
                lower.extend([CombinedHeuristics(combo) for combo in itertools.combinations(h, r=x)])
                [lower_idx.append(combo) for combo in itertools.combinations(range(len(h)), r=x)]

        return upper, lower, upper_idx, lower_idx
