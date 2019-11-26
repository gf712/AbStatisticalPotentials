import os
from glob import glob
import numpy as np
import itertools

from .utils import get_kwargs
from .heuristics import CombinedHeuristics
from .statistical_potential_helper import ApproximateStatisticalPotential
from .math_helpers import inverse_boltzmann
from .io_helpers import serialize_json, deserialize_json
from .heuristics import *

def debug_approx_w(c):
    """Function to get w approximation calculation
    in a readable string format.
    """
    upper = []
    lower = []
    flatten = lambda l: [item for sublist in l for item in sublist]
    for x in range(1, len(c)+1):
        if x % 2 == 1:
            upper.append([f"P({', '.join(i)})" for i in itertools.combinations(c, r=x)])
        else:
            lower.append([f"P({', '.join(i)})" for i in itertools.combinations(c, r=x)])
    return f"({'*'.join(flatten(upper))}) / ({'*'.join(flatten(lower))})"


def debug_compute_w(c):
    """Function to get w calculation
    in a readable string format.
    """
    computations = list()
    for i in range(2, len(c)+1):
        for approx_c_i in itertools.combinations(c, r=i):
            computations.append(debug_approx_w(approx_c_i))
    return ' +\n'.join(computations)



def w_heuristics(h):
    computations = list()
    for i in range(2, len(h) + 1):
        for approx_c_i in itertools.combinations(h, r=i):
            computations.append(ApproximateStatisticalPotential(approx_c_i))
    return computations

class StatisticalPotential:
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError(f"Expected at least two heuristics, but got {len(args)}.")
        self._heuristics = args
        self._approximate_sp = w_heuristics(self._heuristics)
        self._cache = dict()
        self._computation_string = debug_compute_w([x.name for x in self._heuristics])
        self._estimated_inverse_boltzmann = None

    def possible_element_combinations(self):
        return itertools.product(*[h.value_set for h in self._heuristics], repeat=1)

    def estimate_inverse_boltzmann(self):
        """Estimates the inverse Boltzmann for each term in
        the statistical potential estimation expansion.
        See self.computation_string to see what each term represents.
        """
        if self._estimated_inverse_boltzmann is None:
            self._estimated_inverse_boltzmann = estimate_inverse_boltzmann_helper(self, serialize=False)
        return self._estimated_inverse_boltzmann

    def compute_inverse_boltzmann(self, pdbs):
        if self._estimated_inverse_boltzmann is None:
            raise ValueError("Call the `estimate_inverse_boltzmann` method before running `compute_inverse_boltzmann`!")
        return compute_inverse_boltzmann_helper(self, self._estimated_inverse_boltzmann, pdbs)

    def serialize(self, path, set_name):
        """Serialize inverse Boltzmann value estimations
        """
        estimate_inverse_boltzmann_helper(self, serialize=True, path=path, set_name=set_name)
        with open(f"{path}/inverse_boltzmann-{set_name}-info.txt", "w") as f:
            f.write(
                    '\n'.join(h.serialization_name() for h in self._heuristics)
                )

    @staticmethod
    def deserialize(path, set_name, pdbs=None):
        """Deserialize inverse Boltzmann value estimations and
        return a new object with these loaded values.
        """
        files = glob(f"{path}/inverse_boltzmann-*.json")
        info_file = f"{path}/inverse_boltzmann-{set_name}-info.txt"
        if pdbs is None:
            pdbs = list()
        hs = list()
        with open(info_file, 'r') as f:
            for line in f:
                hs.append(line.rstrip())

        hs_instances = list()
        for x in hs:
            hs_instances.append(eval(x.replace("PDBS", "pdbs")))

        sp = StatisticalPotential(*hs_instances)     
        
        boltzmann_values = list()
        for approx_boltzmann in sp._approximate_sp:
            name = approx_boltzmann.name
            filename = f"{path}/inverse_boltzmann-{set_name}-{name}.json" 
            if filename not in files:
                raise ValueError(f"Expected file {filename} not found.")

            boltzmann_values.append(deserialize_json(filename))

        sp._estimated_inverse_boltzmann = boltzmann_values
        
        return sp

    @property
    def computation_string(self):
        return self._computation_string
    
def frequency_computation_helper(subexpression, cache):
    result = list()
    for term in subexpression:
        if term not in cache:                    
            cache[term] = term.compute_frequencies()
        result.append(cache[term])
    return result

def single_struct_frequency_computation_helper(subexpression, idx):
    result = list()
    for term in subexpression:
        result.append(term.compute_frequency(idx))
    return result


def estimate_inverse_boltzmann_helper(statistical_potential, serialize=True, **kwargs):
    result = list()

    for e in statistical_potential._approximate_sp:
        result_i = dict()
        # observed
        upper_computations = frequency_computation_helper(e.upper_computations, 
                                                          statistical_potential._cache)
        # expected
        lower_computations = frequency_computation_helper(e.lower_computations, 
                                                          statistical_potential._cache)

        upper_inverse_boltzmann = dict()
        lower_inverse_boltzmann = dict()

        for combo in e.possible_element_combinations():
            upper_product = 1.0
            lower_product = 1.0
            for upper_term, upper_term_idx in zip(upper_computations, e.upper_idx):
                key = '-'.join(combo[idx] for idx in upper_term_idx)
                if key in upper_term:
                    upper_product *= upper_term[key]
            for lower_term, lower_term_idx in zip(lower_computations, e.lower_idx):
                key = '-'.join(combo[idx] for idx in lower_term_idx)
                if key in lower_term:
                    lower_product *= lower_term[key]

            result_i['-'.join(combo)] = inverse_boltzmann(upper_product, lower_product)

        if serialize:
            directory = get_kwargs('path', kwargs)
            filename = get_kwargs('set_name', kwargs)
            path = f"{directory}/inverse_boltzmann-{filename}-{e.name}.json"
            if not os.path.isdir(directory):
                os.mkdir(directory)
            serialize_json(result_i, path)

        result.append(result_i)

    return result


def compute_inverse_boltzmann_helper(statistical_potential, inverse_boltzmann_values, pdbs):

    new_h = [x.clone_with(pdbs) for x in statistical_potential._heuristics]
    new_sp = StatisticalPotential(*new_h)
    result = list()

    for idx in range(len(pdbs)):
        for e, approx_boltzmann in zip(new_sp._approximate_sp,inverse_boltzmann_values):
            result_i = 0
            hs = CombinedHeuristics(e._heuristics)
            freq = hs.compute_frequency(idx)
            for k, v in freq.items():
                result_i += approx_boltzmann[k] * v
        result.append(result_i)

    return result
