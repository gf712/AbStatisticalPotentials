from scipy import constants
import numpy as np
import math

# https://www.sciencedirect.com/science/article/pii/S1476927114001698
Residue_opt_ref_asa = {
    "ALA": 108.5,
    "ARG": 246.37,
    "ASN": 156.64,
    "ASP": 146.17,
    "CYS": 138.40,
    "GLN": 181.94,
    "GLU": 177.12,
    "GLY": 81.30,
    "HIS": 187.48,
    "ILE": 182.09,
    "LEU": 181.29,
    "LYS": 206.10,
    "MET": 200.54,
    "PHE": 206.76,
    "PRO": 143.86,
    "SER": 119.02,
    "THR": 142.62,
    "TRP": 254.92,
    "TYR": 219.61,
    "VAL": 157.05
}

AA_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
           'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
           'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
           'SER', 'THR', 'TRP', 'TYR', 'VAL']

DIST_CUTOFF = 15 # A
WINDOW_SIZE = 17
BOLTZMANN_CONSTANT = constants.Boltzmann
# from https://www.tandfonline.com/doi/pdf/10.1080/07391102.2015.1073631
DIST_BINS = np.append(np.arange(3, 8.2, 0.2), (np.inf,)) 
SASA_BINS = [0.05, 0.2, 0.4, 0.55, math.inf]
