from glob import glob
import math
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy import constants
import mdtraj as md
import json
import itertools
from abc import ABC, abstractmethod
import copy

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


def to_probabilities(dict_a):
    """Converts a dictionary with observed frequencies to 
    normalised probabilities.
    """
    total = float(sum(dict_a.values()))
    return {k: v/total for k, v in dict_a.items()}


def compute_dssp(pdb):
    """Computes DSSP using MDTraj. Expects a single
    trajectory.
    """
    dssp = md.compute_dssp(pdb, simplified=False)[0]
    # replace blanks with L to make results more readable
    return ['L' if el==' ' else el for el in dssp]


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


def compute_sidechain_pdist(pdb):
    """Computes the pairwise euclidean distance of 
    the geometrical centroid of each amino acid sidechain.
    """
    sidechain_centroids = np.zeros(shape=(pdb.n_residues, 3))
    for i, res in enumerate(pdb.top.residues):
        sidechain_coords = np.array([pdb.xyz[0, atom.index] * 10 for atom in res.atoms if atom.is_sidechain])
        if sidechain_coords.size < 1:
            continue
        sidechain_centroids[i] = sidechain_coords.mean(0)
    sidechain_pdist = squareform(pdist(sidechain_centroids))

    return sidechain_pdist

def serialize_numpy_to_string(array):
    result = []
    for val in array:
        if val == np.inf:
            result.append("np.inf")
        else:
            result.append(str(val))
    result[0] = "np.array([" + result[0]
    result[-1] = result[-1] + "])"
    return ','.join(result)


class Heuristic(ABC):
    """A stateful class that computes a given heuristic
    for all given mdtraj.Trajectory objects.
    """
    def __init__(self, name, pdbs, position, value_set):
        if position not in ['i', 'j', 'ij']:
            raise ValueError("Expeceted position to be either i, j or ij.")
        self._name = name
        self._pdbs = pdbs
        # using list as cache (instead of dict) as hash(pdb) does not always work
        self._position = position
        self._value_set = value_set
        self._initialize()
    
    @abstractmethod
    def _compute(self, pdb):
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def serialization_name(self):
        raise NotImplementedError("Abstract method")

    def _initialize(self):
        self._count = len(self._pdbs)
        self.cache = [None] * len(self._pdbs)

    def compute(self, i):
        if self.cache[i] is None:
            result = self._compute(self._pdbs[i])
            self.cache[i] = result
        return self.cache[i]

    def compute_all(self):
        result = list()
        for i in range(len(self._pdbs)):
            result.append(self.compute(i))
        return result

    def clone_with(self, pdbs):
        """Clones the object with using the same attributes
        but replaces with another set of pdb structures and
        resets the state, e.g. new cache
        """
        new_obj = copy.copy(self)
        new_obj.__dict__['_pdbs'] = pdbs
        new_obj._initialize()
        return new_obj

    @property
    def name(self):
        return self._name + self._position

    @property
    def position(self):
        return self._position

    @property
    def count(self):
        return self._count

    @property
    def value_set(self):
        return self._value_set

    
class SidechainPdistHeuristic(Heuristic):

    def __init__(self, name, pdbs, bins, cutoff):
        super().__init__(name, pdbs, "ij", [str(x) for x in range(len(bins))])
        self._bins = bins
        self._cutoff = cutoff

    def _compute(self, pdb):
        pdist = compute_sidechain_pdist(pdb)
        result = np.full_like(pdist, np.NaN, dtype="<U3")
        for i in range(pdist.shape[0]):
            for j in range(i+1, pdist.shape[1]):
                if pdist[i, j] < self._cutoff:
                    result[i, j] = compute_bin(pdist[i, j], self._bins)
                    result[j, i] = result[i, j]
        return result

    def serialization_name(self):
        return f"SidechainPdistHeuristic('{self._name}', PDBS, {serialize_numpy_to_string(self._bins)}, {self._cutoff})"


class DSSPHeuristic(Heuristic):

    def __init__(self, name, pdbs, cutoff, position):
        super().__init__(name, pdbs, position, ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'L'])
        self._cutoff = cutoff

    def _compute(self, pdb):
        dssp = compute_dssp(pdb)
        result = np.full((len(dssp), len(dssp)), np.NaN, dtype="<U3")
        for i in range(result.shape[0]):
            lower_bound = max(0, i-self._cutoff)
            upper_bound = min(i+self._cutoff, len(dssp))
            for j in range(lower_bound, upper_bound):
                result[i, j] = dssp[j]
            result[i, i] = dssp[i]

        return result

    def serialization_name(self):
        return f"DSSPHeuristic('{self._name}', PDBS, {self._cutoff}, '{self._position}')"

class SASAHeuristic(Heuristic):

    def __init__(self, name, pdbs, bins, cutoff, position):
        super().__init__(name, pdbs, position, [str(x) for x in range(len(bins))])
        self._cutoff = cutoff

    def _compute(self, pdb):
        sasa = md.shrake_rupley(pdb, mode='residue')[0] * 100
        result = np.full((len(sasa), len(sasa)), np.NaN, dtype="<U3")
        resnames = [res.name for res in pdb.residues]
        for i in range(result.shape[0]):
            lower_bound = max(0, i-self._cutoff)
            upper_bound = min(i+self._cutoff, len(dssp))
            for j in range(lower_bound, upper_bound):
                ratio = compute_bin(sasa[j] / Residue_opt_ref_asa[resnames[j]], self._bins)
                bin_i = compute_bin(ratio, bins)
                result[i, j] = ratio
            result[i, i] = compute_bin(sasa[i] / Residue_opt_ref_asa[resnames[i]], self._bins)

        return result

    def serialization_name(self):
        return f"SASAHeuristic('{self._name}', PDBS, {serialize_numpy_to_string(self._bins)}, {self._cutoff}, '{self._position}')"


class AAHeuristic(Heuristic):

    def __init__(self, name, pdbs, cutoff, position):
        super().__init__(name, pdbs, position, AA_LIST)
        self._cutoff = cutoff

    def _compute(self, pdb):
        resnames = [res.name for res in pdb.top.residues]
        result = np.full((len(resnames), len(resnames)), np.NaN, dtype="<U3")
        for i in range(result.shape[0]):
            lower_bound = max(0, i-self._cutoff)
            upper_bound = min(i+self._cutoff, len(resnames))
            for j in range(lower_bound, upper_bound):
                result[i, j] = resnames[j]
            result[i, i] = resnames[i]
        return result

    def serialization_name(self):
        return f"AAHeuristic('{self._name}', PDBS, {self._cutoff}, {self._position})"


class CombinedHeuristics:
    def __init__(self, heuristics):
        count = set(h.count for h in heuristics)
        if len(count) != 1:
            raise ValueError("Expected all heuristic objects to own the same number of structs")
        self._heuristics = heuristics
        self._count = count.pop()
        self._cache = [None] * self._count

    def compute_frequency(self, idx):
        result = defaultdict(int)

        if self._cache[idx] is None:
            h_results = [h.compute(idx) for h in self._heuristics]
            for i in range(h_results[0].shape[0]):
                for j in range(i+1, h_results[0].shape[0]):
                    failed = False
                    key = []
                    for h_i, h in enumerate(self._heuristics):
                        # check if res_i and res_j are available for this computation
                        val_ij = h_results[h_i][i, j]
                        if is_nan(val_ij):
                            failed = True
                            break
                        # if available we can store the value of i or j or ij 
                        # depending on heuristic
                        if h.position == 'i':
                            key.append(str(h_results[h_i][i, i]))
                        else:
                            key.append(val_ij)

                    if not failed:
                        result['-'.join(key)] += 1
            self._cache[idx] = result
        return self._cache[idx]

    def compute_frequencies(self):
        result_list = list()
        result = defaultdict(int)
        for struct_i in range(self._count):
            result_list.append(self.compute_frequency(struct_i))
        for result_i in result_list:
            for k, v in result_i.items():
                result[k] += v
        return {k: float(v) / self._count for k, v in result.items()}

    def clone_with(self, pdbs):
        """Clones CombinedHeuristics which uses
        pdbs internally.
        """
        new_h = [h.clone_with(pdbs) for h in self._heuristics]
        return CombinedHeuristics(new_h)

    def serialize(self):
        """Serialize object
        """
        raise NotImplementedError("serialize not implemented")

    @staticmethod
    def deserialize(path):
        raise NotImplementedError("deserialize not implemented")

    @property
    def name(self):
        return f"CombinedHeuristics({', '.join([h.name for h in self._heuristics])})"
    
    def __repr__(self):
        return self.name

    def __hash__(self):
        """The hash of a CombinedHeuristics object is
        the hash of the string which combines the name
        of all Heuristic objects.
        """
        return hash(''.join([h.name for h in self._heuristics]))

    def __eq__(self, other):
        if not isinstance(other, CombinedHeuristics):
            raise ValueError("Expected another CombinedHeuristics object.")
        return all(this_name == other_name for this_name, other_name in zip([h.name for h in self._heuristics],
            [h.name for h in other._heuristics]))


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

def w_heuristics(h):
    computations = list()
    for i in range(2, len(h) + 1):
        for approx_c_i in itertools.combinations(h, r=i):
            computations.append(ApproximateStatisticalPotential(approx_c_i))
    return computations


def is_nan(val):
    if np.isreal(val):
        return np.isnan(val)
    return val == 'nan'


def get_kwargs(name, kwargs):
    if name not in kwargs:
        raise ValueError(f"Expected {name} in kwargs.")
    return kwargs[name]

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
            path = f"{get_kwargs('path', kwargs)}/inverse_boltzmann-{get_kwargs('set_name', kwargs)}-{e.name}.json"
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
    def deserialize(path, set_name, pdbs):
        """Deserialize inverse Boltzmann value estimations and
        return a new object with these loaded values.
        """
        files = glob(f"{path}/inverse_boltzmann-*.json")
        info_file = f"{path}/inverse_boltzmann-{set_name}-info.txt"
        hs = list()
        with open(info_file, 'r') as f:
            for line in f:
                hs.append(line.rstrip())

        hs_instances = list()
        for x in hs:
            print(x.replace("PDBS", "pdbs"))
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
    

# All the routines for approximating the statistical potentials of a dataset and
# the corresponding functions to infer the energetic contribution of any protein


def w1_calculation(pdbs):
    cs = defaultdict(int)
    c = defaultdict(int)
    s = defaultdict(int)

    for pdb in pdbs:
        dssp = compute_dssp(pdb)
        for res, conformation in zip(pdb.top.residues, dssp):
            if conformation == 'NA':
                continue
            resname = res.name
            c[f"{conformation}"] += 1
            s[f"{resname}"] += 1
            cs[f"{resname}-{conformation}"] += 1
    return cs, c, s


def w1_inference(pdbs, names, cs, c, s):
    result = dict()
    for name, pdb in zip(names, pdbs):
        result_i = 0
        dssp = compute_dssp(pdb)
        for res, conformation in zip(pdb.top.residues, dssp):
            if conformation == 'NA':
                continue
            resname = res.name
            try:
                result_i += inverse_boltzmann(cs[f"{resname}-{conformation}"],
                                              c[conformation], s[resname])
            except:
                pass
        result[name] = result_i
    return result


def w2_calculation(pdbs, bins=SASA_BINS):
    result_hisi_s = defaultdict(int)
    result_hisi_cs = defaultdict(int)
    result_hisi_c = defaultdict(int)

    result_hihjsi_s = defaultdict(int)
    result_hihjsi_cs = defaultdict(int)
    result_hihjsi_c = defaultdict(int)

    result_hisisj_s = defaultdict(int)
    result_hisisj_cs = defaultdict(int)
    result_hisisj_c = defaultdict(int)

    for pdb in pdbs:
        sasa = md.shrake_rupley(pdb, mode='residue')[0] * 100 # nm -> A
        resnames = [res.name for res in pdb.top.residues]

        for i in range(pdb.n_residues):
            if sasa[i] == 0:
                continue
            resname_i = resnames[i]
            ratio = sasa[i] / Residue_opt_ref_asa[resname_i]

            bin_i = compute_bin(ratio, bins)

            result_hisi_s[f"{resname_i}"] += 1
            result_hisi_c[f"{bin_i}"] += 1
            result_hisi_cs[f"{resname_i}-{bin_i}"] += 1
            
            lower_bound = max(0, i-(WINDOW_SIZE//2))
            upper_bound = min(i+(WINDOW_SIZE//2), pdb.n_residues)
            for j in range(lower_bound, upper_bound):
                if sasa[j] == 0:
                    continue
                resname_j = resnames[j]
                ratio_j = sasa[j] / Residue_opt_ref_asa[resname_j]
                bin_j = compute_bin(ratio_j, bins)

                result_hihjsi_cs[f"{resname_i}-{bin_j}-{bin_j}"] += 1
                result_hihjsi_c[f"{bin_i}-{bin_j}"] += 1
                result_hihjsi_s[f"{resname_i}"] += 1

                result_hisisj_cs[f"{resname_i}-{resname_j}-{bin_j}"] +=1 
                result_hisisj_c[f"{bin_j}"] += 1
                result_hisisj_s[f"{resname_i}-{resname_j}"] += 1

    return result_hisi_cs, result_hisi_c, result_hisi_s, \
           result_hihjsi_cs, result_hihjsi_c, result_hihjsi_s,\
           result_hisisj_cs, result_hisisj_c, result_hisisj_s


def w2_inference(pdbs, names,
                 hisi_cs, hisi_c, hisi_s,
                 hihjsi_cs, hihjsi_c, hihjsi_s,
                 hisisj_cs, hisisj_c, hisisj_s,
                 bins=SASA_BINS):
    result = dict()
    for pdb, name in zip(pdbs, names):
        sasa = md.shrake_rupley(pdb, mode='residue')[0] * 100 # nm -> A
        resnames = [res.name for res in pdb.top.residues]
        result_i = 0

        for i in range(pdb.n_residues):
            if sasa[i] == 0:
                continue
            resname_i = resnames[i]
            ratio = sasa[i] / Residue_opt_ref_asa[resname_i]
            bin_i = compute_bin(ratio, bins)

            try:
                result_i += inverse_boltzmann(hisi_cs[f"{resname_i}-{bin_i}"],
                                              hisi_c[f"{bin_i}"], hisi_s[f"{resname_i}"])
            except:
                pass
            lower_bound = max(0, i-(WINDOW_SIZE//2))
            upper_bound = min(i+(WINDOW_SIZE//2), pdb.n_residues)
            for j in range(lower_bound, upper_bound):
                if sasa[j] == 0:
                    continue
                resname_j = resnames[j]
                ratio_j = sasa[j] / Residue_opt_ref_asa[resname_j]
                bin_j = compute_bin(ratio_j, bins)

                try:
                    val = inverse_boltzmann(hihjsi_cs[f"{resname_i}-{bin_j}-{bin_j}"],
                                              hihjsi_c[f"{bin_i}-{bin_j}"], 
                                              hihjsi_s[f"{resname_i}"])
                    val += inverse_boltzmann(hisisj_cs[f"{resname_i}-{resname_j}-{bin_j}"],
                                              hisisj_c[f"{bin_j}"], 
                                              hisisj_s[f"{resname_i}-{resname_j}"])
                    result_i += values
                except:
                    continue
        result[name] = result_i
    return result


def w3_calculation(pdbs, dist_bins=DIST_BINS, cutoff=DIST_CUTOFF):
    result_dijsi_cs = defaultdict(int)
    result_dijsi_s = defaultdict(int)
    result_dijsi_c = defaultdict(int)

    result_dijsisj_cs = defaultdict(int)
    result_dijsisj_s = defaultdict(int)
    result_dijsisj_c = defaultdict(int)

    for pdb in pdbs:
        resnames = [res.name for res in pdb.top.residues]
        sidechain_pdist = compute_sidechain_pdist(pdb)

        # go through the upper triangle of the dist matrix
        for res_i in range(pdb.n_residues):
            for res_j in range(res_i + 1, pdb.n_residues):
                dist = sidechain_pdist[res_i, res_j]
                # ignore very distant residue pairs
                if dist > cutoff:
                    continue
                bin_i = compute_bin(dist, dist_bins)
                
                result_dijsi_cs[f"{resnames[res_i]}-{bin_i}"] +=1
                result_dijsi_c[f"{bin_i}"] +=1
                result_dijsi_s[f"{resnames[res_i]}"] +=1

                result_dijsisj_cs[f"{resnames[res_i]}-{resnames[res_j]}-{bin_i}"] += 1
                result_dijsisj_c[f"{bin_i}"] += 1
                result_dijsisj_s[f"{resnames[res_i]}-{resnames[res_j]}"] += 1
    
    return result_dijsi_cs, result_dijsi_c, result_dijsi_s, \
           result_dijsisj_cs, result_dijsisj_c, result_dijsisj_s


def w3_inference(pdbs, names, 
                 dijsi_cs, dijsi_c, dijsi_s, 
                 dijsisj_cs, dijsisj_c, dijsisj_s,
                 dist_bins=DIST_BINS, cutoff=DIST_CUTOFF):
    results = dict()
    for pdb, name in zip(pdbs, names):
        result_i = 0
        resnames = [res.name for res in pdb.top.residues]
        sidechain_pdist = compute_sidechain_pdist(pdb)

        # go through the upper triangle of the dist matrix
        for res_i in range(pdb.n_residues):
            for res_j in range(res_i + 1, pdb.n_residues):
                dist = sidechain_pdist[res_i, res_j]
                # ignore very distant residue pairs
                if dist > cutoff:
                    continue
                bin_i = compute_bin(dist, dist_bins)

                try:
                    val = inverse_boltzmann(dijsi_cs[f"{resnames[res_i]}-{bin_i}"],
                                                  dijsi_c[f"{bin_i}"], dijsi_s[f"{resnames[res_i]}"])

                    val += inverse_boltzmann(dijsisj_cs[f"{resnames[res_i]}-{resnames[res_j]}-{bin_i}"],
                                                      dijsisj_c[f"{bin_i}"],
                                                      dijsisj_s[f"{resnames[res_i]}-{resnames[res_j]}"])
                    result_i += val
                except:
                    pass
        results[name] = result_i

    return results


def w4_calculation(pdbs, dist_bins=DIST_BINS, cutoff=DIST_CUTOFF):

    result_tidijsi_cs = defaultdict(int)
    result_tidijsi_c = defaultdict(int)
    result_tidijsi_s = defaultdict(int)

    result_tidijsitj_cs = defaultdict(int)
    result_tidijsitj_c = defaultdict(int)
    result_tidijsitj_s = defaultdict(int)

    for pdb in pdbs:
        resnames = [res.name for res in pdb.top.residues]
        sidechain_pdist = compute_sidechain_pdist(pdb)
        dssp = compute_dssp(pdb)
        
        for i in range(pdb.n_residues):
                
            lower_bound = max(0, i-(WINDOW_SIZE//2))
            upper_bound = min(i+(WINDOW_SIZE//2), pdb.n_residues)
                
            for j in range(lower_bound, upper_bound):
                dist = sidechain_pdist[i, j]
                # ignore very distant residue pairs
                if dist > cutoff:
                    continue

                # ignore unavailable dssp 
                if dssp[i] == 'NA' or dssp[j] == 'NA':
                    continue

                dist_bin_i = compute_bin(dist, dist_bins)

                result_tidijsi_cs[f"{resnames[i]}-{dssp[i]}-{dist_bin_i}"] += 1
                result_tidijsi_c[f"{dssp[i]}-{dist_bin_i}"] += 1
                result_tidijsi_s[f"{resnames[i]}"] += 1

                result_tidijsitj_cs[f"{resnames[i]}-{dssp[i]}-{dist_bin_i}-{dssp[j]}"] += 1
                result_tidijsitj_c[f"{dssp[j]}-{dist_bin_i}-{dssp[j]}"] += 1
                result_tidijsitj_s[f"{resnames[i]}"] += 1

    return result_tidijsi_cs, result_tidijsi_c, result_tidijsi_s, \
           result_tidijsitj_cs, result_tidijsitj_c, result_tidijsitj_s


def w4_inference(pdbs, names, 
                 tidijsi_cs, tidijsi_c, tidijsi_s,
                 tidijsitj_cs, tidijsitj_c, tidijsitj_s,
                 dist_bins=DIST_BINS, cutoff=DIST_CUTOFF):

    results = dict()
    for pdb, name in zip(pdbs, names):
        result_i = 0
        resnames = [res.name for res in pdb.top.residues]
        sidechain_pdist = compute_sidechain_pdist(pdb)
        dssp = compute_dssp(pdb)
        
        for i in range(pdb.n_residues):
                
            lower_bound = max(0, i-(WINDOW_SIZE//2))
            upper_bound = min(i+(WINDOW_SIZE//2), pdb.n_residues)
                
            for j in range(lower_bound, upper_bound):
                dist = sidechain_pdist[i, j]
                # ignore very distant residue pairs
                if dist > cutoff:
                    continue

                # ignore unavailable dssp 
                if dssp[i] == 'NA' or dssp[j] == 'NA':
                    continue

                dist_bin_i = compute_bin(dist, dist_bins)

                try:
                    val = inverse_boltzmann(tidijsi_cs[f"{resnames[i]}-{dssp[i]}-{dist_bin_i}"],
                                                  tidijsi_c[f"{dssp[i]}-{dist_bin_i}"],
                                                  tidijsi_s[f"{resnames[i]}"])

                    val += inverse_boltzmann(tidijsitj_cs[f"{resnames[i]}-{dssp[i]}-{dist_bin_i}-{dssp[j]}"],
                                                  tidijsitj_c[f"{dssp[j]}-{dist_bin_i}-{dssp[j]}"],
                                                  tidijsitj_s[f"{resnames[i]}"])
                    result_i = val
                except:
                    pass
        results[name] = result_i
    return results


def w5_calculation(pdbs, dist_bins=DIST_BINS, cutoff=DIST_CUTOFF, sasa_bins=SASA_BINS):

    result_hidijsi_cs = defaultdict(int)
    result_hidijsi_c = defaultdict(int)
    result_hidijsi_s = defaultdict(int)

    for pdb in pdbs:
        resnames = [res.name for res in pdb.top.residues]
        sidechain_pdist = compute_sidechain_pdist(pdb)
        sasa = md.shrake_rupley(pdb, mode='residue')[0] * 100 # nm -> A

        for i in range(pdb.n_residues):
            if sasa[i] == 0:
                continue
            resname_i = resnames[i]
            ratio = sasa[i] / Residue_opt_ref_asa[resname_i]
            sasa_bin_i = compute_bin(ratio, sasa_bins)
            
            lower_bound = max(0, i-(WINDOW_SIZE//2))
            upper_bound = min(i+(WINDOW_SIZE//2), pdb.n_residues)
        
            for j in range(lower_bound, upper_bound):
                dist = sidechain_pdist[i, j]
                # ignore very distant residue pairs
                if dist > cutoff:
                    continue

                dist_bin_i = compute_bin(dist, dist_bins)

                result_hidijsi_cs[f"{resname_i}-{sasa_bin_i}-{dist_bin_i}"] += 1
                result_hidijsi_c[f"{sasa_bin_i}-{dist_bin_i}"] += 1
                result_hidijsi_s[f"{resname_i}"] += 1

    return result_hidijsi_cs, result_hidijsi_c, result_hidijsi_s


def w5_inference(pdbs, names, 
                 hidijsi_cs, hidijsi_c, hidijsi_s,
                 dist_bins=DIST_BINS, cutoff=DIST_CUTOFF, sasa_bins=SASA_BINS):
    
    results = dict()
    for pdb, name in zip(pdbs, names):
        result_i = 0
        resnames = [res.name for res in pdb.top.residues]
        sidechain_pdist = compute_sidechain_pdist(pdb)
        sasa = md.shrake_rupley(pdb, mode='residue')[0] * 100 # nm -> A

        for i in range(pdb.n_residues):
            if sasa[i] == 0:
                continue
            resname_i = resnames[i]
            ratio = sasa[i] / Residue_opt_ref_asa[resname_i]
            sasa_bin_i = compute_bin(ratio, sasa_bins)
            
            lower_bound = max(0, i-(WINDOW_SIZE//2))
            upper_bound = min(i+(WINDOW_SIZE//2), pdb.n_residues)
            
            for j in range(lower_bound, upper_bound):
                dist = sidechain_pdist[i, j]
                # ignore very distant residue pairs
                if dist > cutoff:
                    continue

                dist_bin_i = compute_bin(dist, dist_bins)

                try:
                    result_i += inverse_boltzmann(hidijsi_cs[f"{resname_i}-{sasa_bin_i}-{dist_bin_i}"],
                                                  hidijsi_c[f"{sasa_bin_i}-{dist_bin_i}"],
                                                  hidijsi_s[f"{resname_i}"])
                except:
                    pass
        results[name] = result_i
    return results


def w6_calculation(pdbs, sasa_bins=SASA_BINS):
    result_hitisi_cs = defaultdict(int)
    result_hitisi_c = defaultdict(int)
    result_hitisi_s = defaultdict(int)

    for pdb in pdbs:
        resnames = [res.name for res in pdb.top.residues]
        sasa = md.shrake_rupley(pdb, mode='residue')[0] * 100 # nm -> A
        dssp = compute_dssp(pdb)

        for i in range(pdb.n_residues):
            if sasa[i] == 0:
                continue
            if dssp[i] == 'NA':
                continue
            resname_i = resnames[i]
            ratio = sasa[i] / Residue_opt_ref_asa[resname_i]
            sasa_bin_i = compute_bin(ratio, sasa_bins)

            result_hitisi_cs[f"{resnames[i]}-{sasa_bin_i}-{dssp[i]}"] += 1
            result_hitisi_c[f"{sasa_bin_i}-{dssp[i]}"] += 1
            result_hitisi_s[f"{resnames[i]}"] += 1

    return result_hitisi_cs, result_hitisi_c, result_hitisi_s


def w6_inference(pdbs, names,
                 hitisi_cs, hitisi_c, hitisi_s,
                 sasa_bins=SASA_BINS):
    results = dict()
    for pdb, name in zip(pdbs, names):
        result_i = 0
        resnames = [res.name for res in pdb.top.residues]
        sasa = md.shrake_rupley(pdb, mode='residue')[0] * 100 # nm -> A
        dssp = compute_dssp(pdb)

        for i in range(pdb.n_residues):
            if sasa[i] == 0:
                continue
            if dssp[i] == 'NA':
                continue
            resname_i = resnames[i]
            ratio = sasa[i] / Residue_opt_ref_asa[resname_i]
            sasa_bin_i = compute_bin(ratio, sasa_bins)

            try:
                result_i += inverse_boltzmann(hitisi_cs[f"{resnames[i]}-{sasa_bin_i}-{dssp[i]}"],
                                              hitisi_c[f"{sasa_bin_i}-{dssp[i]}"],
                                              hitisi_s[f"{resnames[i]}"])
            except:
                pass
        results[name] = result_i
    return results


def serialize_json(var, name):
    """Helper function to serialise a Python dictionary in
    JSON format. Handles opening and closing file.
    """
    with open(name, "w") as f:
        json.dump(var, f)


def deserialize_json(path):
    result = {}
    with open(path, "r") as f:
        result = json.load(f)
    return result


def compute_statistical_potential(pdbs, dataset_path, dataset_name):
    """The main routine to compute the statistical potential of dataset.
    The results are serealised in JSON format.
    """

    results = dict()
    
    w1 = w1_calculation(pdbs)
    w2 = w2_calculation(pdbs)
    w3 = w3_calculation(pdbs)
    w4 = w4_calculation(pdbs)
    w5 = w5_calculation(pdbs)
    w6 = w6_calculation(pdbs)

    results["W1_F_TERM1_CS"] = to_probabilities(w1[0])
    results["W1_F_TERM1_C"]  = to_probabilities(w1[1])
    results["W1_F_TERM1_S"]  = to_probabilities(w1[2])

    results["W2_F_TERM1_CS"] = to_probabilities(w2[0])
    results["W2_F_TERM1_C"]  = to_probabilities(w2[1])
    results["W2_F_TERM1_S"]  = to_probabilities(w2[2])
    results["W2_F_TERM2_CS"] = to_probabilities(w2[3])
    results["W2_F_TERM2_C"]  = to_probabilities(w2[4])
    results["W2_F_TERM2_S"]  = to_probabilities(w2[5])
    results["W2_F_TERM3_CS"] = to_probabilities(w2[6])
    results["W2_F_TERM3_C"]  = to_probabilities(w2[7])
    results["W2_F_TERM3_S"]  = to_probabilities(w2[8])

    results["W3_F_TERM1_CS"] = to_probabilities(w3[0])
    results["W3_F_TERM1_C"]  = to_probabilities(w3[1])
    results["W3_F_TERM1_S"]  = to_probabilities(w3[2])
    results["W3_F_TERM2_CS"] = to_probabilities(w3[3])
    results["W3_F_TERM2_C"]  = to_probabilities(w3[4])
    results["W3_F_TERM2_S"]  = to_probabilities(w3[5])

    results["W4_F_TERM1_CS"] = to_probabilities(w4[0])
    results["W4_F_TERM1_C"]  = to_probabilities(w4[1])
    results["W4_F_TERM1_S"]  = to_probabilities(w4[2])
    results["W4_F_TERM2_CS"] = to_probabilities(w4[3])
    results["W4_F_TERM2_C"]  = to_probabilities(w4[4])
    results["W4_F_TERM2_S"]  = to_probabilities(w4[5])

    results["W5_F_TERM1_CS"] = to_probabilities(w5[0])
    results["W5_F_TERM1_C"]  = to_probabilities(w5[1])
    results["W5_F_TERM1_S"]  = to_probabilities(w5[2])

    results["W6_F_TERM1_CS"] = to_probabilities(w6[0])
    results["W6_F_TERM1_C"]  = to_probabilities(w6[1])
    results["W6_F_TERM1_S"]  = to_probabilities(w6[2])
    
    for name, w in results.items():
        serialize_json(w, f"{dataset_path}/{name}-{dataset_name}.json")


def get_statistical_potentials(dataset_path, dataset_name):
    result = dict()
    result["W1_F_TERM1_CS"] = deserialize_json(f"{dataset_path}/W1_F_TERM1_CS-{dataset_name}.json")
    result["W1_F_TERM1_C"]  = deserialize_json(f"{dataset_path}/W1_F_TERM1_C-{dataset_name}.json")
    result["W1_F_TERM1_S"]  = deserialize_json(f"{dataset_path}/W1_F_TERM1_S-{dataset_name}.json")
    
    result["W2_F_TERM1_CS"] = deserialize_json(f"{dataset_path}/W2_F_TERM1_CS-{dataset_name}.json")
    result["W2_F_TERM1_C"]  = deserialize_json(f"{dataset_path}/W2_F_TERM1_C-{dataset_name}.json")
    result["W2_F_TERM1_S"]  = deserialize_json(f"{dataset_path}/W2_F_TERM1_S-{dataset_name}.json")
    result["W2_F_TERM2_CS"] = deserialize_json(f"{dataset_path}/W2_F_TERM2_CS-{dataset_name}.json")
    result["W2_F_TERM2_C"]  = deserialize_json(f"{dataset_path}/W2_F_TERM2_C-{dataset_name}.json")
    result["W2_F_TERM2_S"]  = deserialize_json(f"{dataset_path}/W2_F_TERM2_S-{dataset_name}.json")
    result["W2_F_TERM3_CS"] = deserialize_json(f"{dataset_path}/W2_F_TERM3_CS-{dataset_name}.json")
    result["W2_F_TERM3_C"]  = deserialize_json(f"{dataset_path}/W2_F_TERM3_C-{dataset_name}.json")
    result["W2_F_TERM3_S"]  = deserialize_json(f"{dataset_path}/W2_F_TERM3_S-{dataset_name}.json")
    
    result["W3_F_TERM1_CS"] = deserialize_json(f"{dataset_path}/W3_F_TERM1_CS-{dataset_name}.json")
    result["W3_F_TERM1_C"]  = deserialize_json(f"{dataset_path}/W3_F_TERM1_C-{dataset_name}.json")
    result["W3_F_TERM1_S"]  = deserialize_json(f"{dataset_path}/W3_F_TERM1_S-{dataset_name}.json")
    result["W3_F_TERM2_CS"] = deserialize_json(f"{dataset_path}/W3_F_TERM2_CS-{dataset_name}.json")
    result["W3_F_TERM2_C"]  = deserialize_json(f"{dataset_path}/W3_F_TERM2_C-{dataset_name}.json")
    result["W3_F_TERM2_S"]  = deserialize_json(f"{dataset_path}/W3_F_TERM2_S-{dataset_name}.json")
    
    result["W4_F_TERM1_CS"] = deserialize_json(f"{dataset_path}/W4_F_TERM1_CS-{dataset_name}.json")
    result["W4_F_TERM1_C"]  = deserialize_json(f"{dataset_path}/W4_F_TERM1_C-{dataset_name}.json")
    result["W4_F_TERM1_S"]  = deserialize_json(f"{dataset_path}/W4_F_TERM1_S-{dataset_name}.json")
    result["W4_F_TERM2_CS"] = deserialize_json(f"{dataset_path}/W4_F_TERM2_CS-{dataset_name}.json")
    result["W4_F_TERM2_C"]  = deserialize_json(f"{dataset_path}/W4_F_TERM2_C-{dataset_name}.json")
    result["W4_F_TERM2_S"]  = deserialize_json(f"{dataset_path}/W4_F_TERM2_S-{dataset_name}.json")
    
    result["W5_F_TERM1_CS"] = deserialize_json(f"{dataset_path}/W5_F_TERM1_CS-{dataset_name}.json")
    result["W5_F_TERM1_C"]  = deserialize_json(f"{dataset_path}/W5_F_TERM1_C-{dataset_name}.json")
    result["W5_F_TERM1_S"]  = deserialize_json(f"{dataset_path}/W5_F_TERM1_S-{dataset_name}.json")
    
    result["W6_F_TERM1_CS"] = deserialize_json(f"{dataset_path}/W6_F_TERM1_CS-{dataset_name}.json")
    result["W6_F_TERM1_C"]  = deserialize_json(f"{dataset_path}/W6_F_TERM1_C-{dataset_name}.json")
    result["W6_F_TERM1_S"]  = deserialize_json(f"{dataset_path}/W6_F_TERM1_S-{dataset_name}.json")

    return result


def compute_inverse_boltzmann(pdbs, pdb_names, dataset_path, dataset_names):
    result = dict()
    for dataset in dataset_names:
        statistical_potentials = get_statistical_potentials(dataset_path, dataset)
        w1 = w1_inference(pdbs, pdb_names, statistical_potentials["W1_F_TERM1_CS"], 
                          statistical_potentials["W1_F_TERM1_C"], 
                          statistical_potentials["W1_F_TERM1_S"])
        w2 = w2_inference(pdbs, pdb_names, statistical_potentials["W2_F_TERM1_CS"], 
                          statistical_potentials["W2_F_TERM1_C"], statistical_potentials["W2_F_TERM1_S"],
                          statistical_potentials["W2_F_TERM2_CS"], statistical_potentials["W2_F_TERM2_C"], 
                          statistical_potentials["W2_F_TERM2_S"], statistical_potentials["W2_F_TERM3_CS"],
                          statistical_potentials["W2_F_TERM3_C"], statistical_potentials["W2_F_TERM3_S"])
        w3 = w3_inference(pdbs, pdb_names, statistical_potentials["W3_F_TERM1_CS"], statistical_potentials["W3_F_TERM1_C"], 
                          statistical_potentials["W3_F_TERM1_S"], statistical_potentials["W3_F_TERM2_CS"], 
                          statistical_potentials["W3_F_TERM2_C"], statistical_potentials["W3_F_TERM2_S"])
        w4 = w4_inference(pdbs, pdb_names, statistical_potentials["W4_F_TERM1_CS"], statistical_potentials["W4_F_TERM1_C"], 
                          statistical_potentials["W4_F_TERM1_S"], statistical_potentials["W4_F_TERM2_CS"], 
                          statistical_potentials["W4_F_TERM2_C"], statistical_potentials["W4_F_TERM2_S"])
        w5 = w5_inference(pdbs, pdb_names, statistical_potentials["W5_F_TERM1_CS"], statistical_potentials["W5_F_TERM1_C"], 
                          statistical_potentials["W5_F_TERM1_S"])
        w6 = w6_inference(pdbs, pdb_names, statistical_potentials["W6_F_TERM1_CS"], statistical_potentials["W6_F_TERM1_C"], 
                          statistical_potentials["W6_F_TERM1_S"])

        result[f"w1-{dataset}"] = w1
        result[f"w2-{dataset}"] = w2
        result[f"w3-{dataset}"] = w3
        result[f"w4-{dataset}"] = w4
        result[f"w5-{dataset}"] = w5
        result[f"w6-{dataset}"] = w6

    return result
