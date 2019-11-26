import numpy as np
import math 

from .definitions import BOLTZMANN_CONSTANT

def to_probabilities(dict_a):
    """Converts a dictionary with observed frequencies to 
    normalised probabilities.
    """
    total = float(sum(dict_a.values()))
    return {k: v/total for k, v in dict_a.items()}


def inverse_boltzmann(observed, expected):
    """ Inverse Boltzmann law:
    -k * T * log(frac{F(observed)/F(expected)})
    """
    return -BOLTZMANN_CONSTANT * 310.15 * math.log(observed / expected)


def compute_bin(val, bins):
    """Computes the bin in which a value belongs to.
    Note that the bin values correspond to the upper
    bound of the ith bin.
    """
    bin_result = 0
    while (val > bins[bin_result]):
        bin_result += 1
    return bin_result

def is_nan(val):
    if np.isreal(val):
        return np.isnan(val)
    return val == 'nan'
