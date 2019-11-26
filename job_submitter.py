import yaml
import mdtraj as md
from statistical_potentials import SidechainPdistHeuristic, DSSPHeuristic, \
	StatisticalPotential, AAHeuristic, SASAHeuristic
from glob import glob
import os
import argparse
import logging
import math
import copy


def load_yaml(file):
	with open(file, 'r') as f:
		data = yaml.load(f, Loader=yaml.FullLoader)
	return data


def load_datasets(datasets):
	structs = dict()
	for name, dataset in datasets.items():
		if "path" not in dataset:
			raise ValueError(f"Expected 'path' field in {name}")
		path = dataset['path']
		if not os.path.isdir(path):
			raise ValueError(f"The provided path {path} could not be found.")
		files = glob(f"{path}/*pdb")
		if len(files) < 1:
			raise ValueError(f"Could not find any pdb files in {path}.")
		logging.info(f"Loading {len(files)} PDB files from {dataset['path']}")
		structs[name] = [md.load_pdb(file) for file in files]

	return structs


def load_heuristic(h_func, **kwargs):
	if h_func == 'DSSP':
		return DSSPHeuristic(**kwargs)
	if h_func == 'SidechainPairwiseDistance':
		return SidechainPdistHeuristic(**kwargs)
	if h_func == 'AminoAcid':
		return AAHeuristic(**kwargs)
	if h_func == 'SASA':
		return SASAHeuristic(**kwargs)
	raise ValueError(f"Unknown heuristic {h_func}")


def load_heuristics(heuristics, pdbs):
	result = list()
	for name, arguments in heuristics.items():
		arguments['pdbs'] = pdbs
		arguments['name'] = name
		h_func = arguments.pop("heuristic")
		if 'bins' in arguments:
			arguments['bins'].append(math.inf)
		result.append(load_heuristic(h_func, **arguments))
	return result


def main(filename):
	yaml_data = load_yaml(filename)
	if "datasets" not in yaml_data:
		raise ValueError("Expected datasets field in YAML file.")
	datasets = yaml_data["datasets"]

	if "serialization_path" not in yaml_data:
		raise ValueError("Expected serialization_path field in YAML file.")
	serialization_path = yaml_data["serialization_path"]

	pdb_dict = load_datasets(datasets)

	if "heuristics" not in yaml_data:
		raise ValueError("Expected heuristics field in YAML file.")

	heuristics = yaml_data['heuristics']

	if len(heuristics) < 2:
		raise ValueError(f"Expected at least 2 heuristics, but got {len(heuristics)}")

	dataset_jobs = dict()
	for name, dataset_pdbs in pdb_dict.items():
		heuristics_copy = copy.deepcopy(heuristics)
		dataset_jobs[name] = StatisticalPotential(*load_heuristics(heuristics_copy, dataset_pdbs))

	for name, jobs in dataset_jobs.items():
		logging.info(f"Processing dataset '{name}'")
		jobs.serialize(serialization_path, name)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='FOO')

	parser.add_argument('filename', type=str, nargs=1,
                    help='YAML filename')

	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO)
	
	main(args.filename[0])
