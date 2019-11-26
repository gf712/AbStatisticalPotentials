A python library to calculate statistical potentials of proteins using the inverse Boltzmann equation.

The library can be used either with a YAML configuration file or using the python API. The data is serialized in a 
special format that allows a user to load data from a calculation from a python session.

In the backend the library uses MDTraj, but an option for ProStruct will be added (which runs faster, but is not tested as much). The computations are cached and to leverage this the user should reuse Heuristic objects when relevant, e.g. 
two StatisticalPotential objects using DSSPHeuristic for the same position and PDB dataset.

## TODO:
	- Add support for sparse data correction
	- Add support for calculating various statistical potential values using the job_submitter.py
	- Add parallel processing option
	- Add optimisation where a user can manually share StatisticalPotential heuristics for efficient caching 

## Usage:

### Estimate the inverse Bolztmann values

#### Using a YAML configuration file:
```yaml
datasets:
  dataset_name:
    path: /some/path/to/directory/with/pdbs
  ...

heuristics:
  ti:
    heuristic: DSSP
    cutoff: 8
    position: i
  tj:
    heuristic: DSSP
    cutoff: 8
    position: j
  dij:
    heuristic: SidechainPairwiseDistance
    cutoff: 10
    bins: [3. , 3.2, 3.4, 3.6, 3.8, 4. , 4.2, 4.4, 4.6, 4.8, 5. , 5.2, 5.4, 5.6, 5.8, 6., 6.2, 6.4, 6.6, 6.8, 7. , 7.2, 7.4, 7.6, 7.8, 8. ]

serialization_path: /directory/where/paths/will/be/stored
```
```bash
python job_submitter.py my_config.yml 
```

#### Using the Python API:
```python
>>> import mdtraj as md
>>> inference_pdbs = [md.load_pdb(file) for file in files]
>>> # define heuristics
>>> h1 = SidechainPdistHeuristic("pdist", inference_pdbs, DIST_BINS, 10)
>>> h2 = DSSPHeuristic("dssp", inference_pdbs, 8, 'i')
>>> h3 = DSSPHeuristic("dssp", inference_pdbs, 8, 'j')
>>> # combine heuristics that define the statistical potential
>>> sp = StatisticalPotential(h1, h2, h3)
>>> # estimate the inverse Boltzmann values using a provided set
>>> sp.estimate_inverse_boltzmann()
>>> # serialize data
>>> sp.serialize(path, dataset_name)
```

### Import and use computed values
```python
>>> from statistical_potential import StatisticalPotential
>>> import mdtraj as md
>>> # load data from the computation above
>>> sp = StatisticalPotential.deserialize(path, dataset_name)
>>> # load a set of structures for inverse Boltzmann estimation
>>> prediction_set = [md.load_pdb(file) for file in files]
>>> # compute inverse Boltzmann on new set with the values estimated from the job above
>>> sp.compute_inverse_boltzmann(prediction_set)
