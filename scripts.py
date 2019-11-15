import math
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy import constants
import mdtraj as md
import json

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

def inverse_boltzmann(frequency_cs, frequency_c, frequency_s):
    """ Inverse Boltzmann law:
    -k * T * log(frac{F(c,s), F(s) * F(c)})
    """
    return -BOLTZMANN_CONSTANT * 310.15 * math.log(
            frequency_cs / (frequency_c * frequency_s)
        )

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
    # square form is more convenient to work with
    sidechain_pdist = squareform(pdist(sidechain_centroids))

    return sidechain_pdist

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
            result_i += inverse_boltzmann(cs[f"{resname}-{conformation}"],
                                          (c[conformation], s[resname]))
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

            result_i += inverse_boltzmann(hisi_cs[f"{resname_i}-{bin_i}"],
                                          hisi_c[f"{bin_i}"], hisi_s[f"{resname_i}"])

            lower_bound = max(0, i-(WINDOW_SIZE//2))
            upper_bound = min(i+(WINDOW_SIZE//2), pdb.n_residues)
            for j in range(lower_bound, upper_bound):
                if sasa[j] == 0:
                    continue
                resname_j = resnames[j]
                ratio_j = sasa[j] / Residue_opt_ref_asa[resname_j]
                bin_j = compute_bin(ratio_j, bins)

                result_i += inverse_boltzmann(hihjsi_cs[f"{resname_i}-{bin_j}-{bin_j}"],
                                              hihjsi_c[f"{bin_i}-{bin_j}"], 
                                              hihjsi_s[f"{resname_i}"])
                result_i += inverse_boltzmann(hisisj_cs[f"{resname_i}-{resname_j}-{bin_j}"],
                                              hisisj_c[f"{bin_j}"], 
                                              hisisj_s[f"{resname_i}-{resname_j}"])
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

                result_i += inverse_boltzmann(dijsi_cs[f"{resnames[res_i]}-{bin_i}"],
                                              dijsi_c[f"{bin_i}"], dijsi_s[f"{resnames[res_i]}"])

                result_i += inverse_boltzmann(dijsisj_cs[f"{resnames[res_i]}-{resnames[res_j]}-{bin_i}"],
                                              dijsisj_c[f"{bin_i}"],
                                              dijsisj_s[f"{resnames[res_i]}-{resnames[res_j]}"])
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

                result_i += inverse_boltzmann(tidijsi_cs[f"{resnames[i]}-{dssp[i]}-{dist_bin_i}"],
                                              tidijsi_c[f"{dssp[i]}-{dist_bin_i}"],
                                              tidijsi_s[f"{resnames[i]}"])

                result_i += inverse_boltzmann(tidijsitj_cs[f"{resnames[i]}-{dssp[i]}-{dist_bin_i}-{dssp[j]}"],
                                              tidijsitj_c[f"{dssp[j]}-{dist_bin_i}-{dssp[j]}"],
                                              tidijsitj_s[f"{resnames[i]}"])
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

                result_i += inverse_boltzmann(hidijsi_cs[f"{resname_i}-{sasa_bin_i}-{dist_bin_i}"],
                                              hidijsi_c[f"{sasa_bin_i}-{dist_bin_i}"],
                                              hidijsi_s[f"{resname_i}"])
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

            result_i += inverse_boltzmann(hitisi_cs[f"{resnames[i]}-{sasa_bin_i}-{dssp[i]}"],
                                          hitisi_c[f"{sasa_bin_i}-{dssp[i]}"],
                                          hitisi_s[f"{resnames[i]}"])
        results[name] = result_i
    return results

def serialize(var, name):
    """Helper function to serialise a Python dictionary in
    JSON format. Handles opening and closing file.
    """
    with open(name, "w") as f:
        json.dump(var, f)
    
def compute_statistical_potential(pdbs, dataset_path, dataset_name):
    """The main routine to compute the statistical potential of dataset.
    The results are serealised in JSON format.
    """
    w1 = w1_calculation(pdbs)
    w2 = w2_calculation(pdbs)
    w3 = w3_calculation(pdbs)
    w4 = w4_calculation(pdbs)
    w5 = w5_calculation(pdbs)
    w6 = w6_calculation(pdbs)
    
    W1_F_TERM1_CS = to_probabilities(w1[0])
    W1_F_TERM1_C = to_probabilities(w1[1])
    W1_F_TERM1_S = to_probabilities(w1[2])

    W2_F_TERM1_CS = to_probabilities(w2[0])
    W2_F_TERM1_C = to_probabilities(w2[1])
    W2_F_TERM1_S = to_probabilities(w2[2])
    W2_F_TERM2_CS = to_probabilities(w2[3])
    W2_F_TERM2_C = to_probabilities(w2[4])
    W2_F_TERM2_S = to_probabilities(w2[5])
    W2_F_TERM3_CS = to_probabilities(w2[6])
    W2_F_TERM3_C = to_probabilities(w2[7])
    W2_F_TERM3_S = to_probabilities(w2[8])

    W3_F_TERM1_CS = to_probabilities(w3[0])
    W3_F_TERM1_C = to_probabilities(w3[1])
    W3_F_TERM1_S = to_probabilities(w3[2])
    W3_F_TERM2_CS = to_probabilities(w3[3])
    W3_F_TERM2_C = to_probabilities(w3[4])
    W3_F_TERM2_S = to_probabilities(w3[5])

    W4_F_TERM1_CS =  to_probabilities(w4[0])
    W4_F_TERM1_C = to_probabilities(w4[1])
    W4_F_TERM1_S =  to_probabilities(w4[2])
    W4_F_TERM2_CS =  to_probabilities(w4[3])
    W4_F_TERM2_C = to_probabilities(w4[4])
    W4_F_TERM2_S =  to_probabilities(w4[5])

    W5_F_TERM1_CS =  to_probabilities(w5[0])
    W5_F_TERM1_C = to_probabilities(w5[1])
    W5_F_TERM1_S =  to_probabilities(w5[2])

    W6_F_TERM1_CS =  to_probabilities(w6[0])
    W6_F_TERM1_C = to_probabilities(w6[1])
    W6_F_TERM1_S =  to_probabilities(w6[2])
    
    weights = [
        W1_F_TERM1_CS,
        W1_F_TERM1_C ,
        W1_F_TERM1_S ,
        W2_F_TERM1_CS,
        W2_F_TERM1_C ,
        W2_F_TERM1_S ,
        W2_F_TERM2_CS,
        W2_F_TERM2_C ,
        W2_F_TERM2_S ,
        W2_F_TERM3_CS,
        W2_F_TERM3_C ,
        W2_F_TERM3_S ,
        W3_F_TERM1_CS,
        W3_F_TERM1_C ,
        W3_F_TERM1_S ,
        W3_F_TERM2_CS,
        W3_F_TERM2_C ,
        W3_F_TERM2_S ,
        W4_F_TERM1_CS,
        W4_F_TERM1_C ,
        W4_F_TERM1_S ,
        W4_F_TERM2_CS,
        W4_F_TERM2_C ,
        W4_F_TERM2_S ,
        W5_F_TERM1_CS,
        W5_F_TERM1_C ,
        W5_F_TERM1_S ,
        W6_F_TERM1_CS,
        W6_F_TERM1_C ,
        W6_F_TERM1_S 
    ]
    weight_names = [
        "W1_F_TERM1_CS",
        "W1_F_TERM1_C", 
        "W1_F_TERM1_S", 
        "W2_F_TERM1_CS",
        "W2_F_TERM1_C", 
        "W2_F_TERM1_S", 
        "W2_F_TERM2_CS",
        "W2_F_TERM2_C",
        "W2_F_TERM2_S",
        "W2_F_TERM3_CS",
        "W2_F_TERM3_C",
        "W2_F_TERM3_S",
        "W3_F_TERM1_CS",
        "W3_F_TERM1_C",
        "W3_F_TERM1_S",
        "W3_F_TERM2_CS",
        "W3_F_TERM2_C",
        "W3_F_TERM2_S",
        "W4_F_TERM1_CS",
        "W4_F_TERM1_C",
        "W4_F_TERM1_S",
        "W4_F_TERM2_CS",
        "W4_F_TERM2_C",
        "W4_F_TERM2_S",
        "W5_F_TERM1_CS",
        "W5_F_TERM1_C",
        "W5_F_TERM1_S",
        "W6_F_TERM1_CS",
        "W6_F_TERM1_C",
        "W6_F_TERM1_S"
    ]
    
    for w, name in zip(weights, weight_names):
        serialize(w, f"{dataset_path}/{name}-{dataset_name}.json")
