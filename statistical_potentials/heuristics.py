from abc import ABC, abstractmethod
import mdtraj as md
import numpy as np
from collections import defaultdict
import copy

from .heuristic_helpers import compute_dssp, compute_sidechain_pdist
from .math_helpers import compute_bin, is_nan
from .io_helpers import serialize_numpy_to_string
from .definitions import *

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


    def __eq__(self, other):
        return (self._name == other._name) and (self._position == other._position) and (self._value_set == other._value_set)


    def __hash__(self):
        # if boost_combine was python code
        hash_value = hash(self._name)
        hash_value ^= hash(self._position) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2)
        hash_value ^= hash(self._value_set) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);

        return hash_value


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
        self._bins = bins

    def _compute(self, pdb):
        sasa = md.shrake_rupley(pdb, mode='residue')[0] * 100
        result = np.full((len(sasa), len(sasa)), np.NaN, dtype="<U3")
        resnames = [res.name for res in pdb.top.residues]
        for i in range(result.shape[0]):
            lower_bound = max(0, i-self._cutoff)
            upper_bound = min(i+self._cutoff, len(result))
            for j in range(lower_bound, upper_bound):
                ratio = compute_bin(sasa[j] / Residue_opt_ref_asa[resnames[j]], self._bins)
                bin_i = compute_bin(ratio, self._bins)
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
        return f"AAHeuristic('{self._name}', PDBS, {self._cutoff}, '{self._position}')"


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
