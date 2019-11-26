import mdtraj as md
import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_dssp(pdb):
    """Computes DSSP using MDTraj. Expects a single
    trajectory.
    """
    dssp = md.compute_dssp(pdb, simplified=False)[0]
    # replace blanks with L to make results more readable
    return ['L' if el==' ' else el for el in dssp]


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