import mdtraj as md
import numpy as np
from glob import glob
import pandas as pd
from statistical_potentials import SidechainPdistHeuristic, DSSPHeuristic, StatisticalPotential
from statistical_potentials.definitions import DIST_BINS


files = glob("/home/gf712/data/Jain_models/*.pdb")
data = pd.read_csv("/home/gf712/data/Jain_models/jain2017_all.csv", index_col=0)
names = [file.split('/')[-1].split('.')[0] for file in files]

mesostable = [name for name in names if data.loc[name]['Tm50'] <= 70.0][:5]
thermostable = [name for name in names if data.loc[name]['Tm50'] > 70.0][:5]

mesostable_pdbs = [md.load_pdb(file) for file in files if file.split('/')[-1].split('.')[0] in mesostable]
thermostable_pdbs = [md.load_pdb(file) for file in files if file.split('/')[-1].split('.')[0] in thermostable]

DIST_BINS = np.append(np.arange(3, 8.2, 0.2), (np.inf,)) 

h1 = SidechainPdistHeuristic("pdist", mesostable_pdbs, DIST_BINS, 10)
h2 = DSSPHeuristic("dssp", mesostable_pdbs, 8, 'i')
h3 = DSSPHeuristic("dssp", mesostable_pdbs, 8, 'j')

h1h2h3 = StatisticalPotential(h1, h2, h3)

print(h1h2h3.computation_string)

r=h1h2h3.estimate_inverse_boltzmann()
h1h2h3.serialize("./test", "test_set")
h1h2h3_copy = h1h2h3.deserialize("./test", "test_set", mesostable_pdbs)

for val, val_from_copy in zip(r, h1h2h3_copy.estimate_inverse_boltzmann()):
	assert val == val_from_copy