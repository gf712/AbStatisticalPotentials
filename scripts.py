import json
import numpy as np
import mdtraj as md
from statistical_potentials.definitions import *
from statistical_potentials.io_helpers import deserialize_json, serialize_json
from statistical_potentials.heuristic_helpers import *
from statistical_potentials.math_helpers import compute_bin


def inverse_boltzmann(a, b, c):
    return -BOLTZMANN_CONSTANT * 310.15 * math.log(a * b / c)



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
