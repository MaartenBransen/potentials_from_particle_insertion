# potentials_from_particle_insertion
## Overview
Python package with a set of functions for calculating the g(r) using the Widom test-particle insertion method, and/or for fitting the g(r) to solve for the pair-potential using particle coordinates from e.g. confocal microscopy

## Installation

### PIP
This package can be installed directly from GitHub using pip:
```
pip install git+https://github.com/MaartenBransen/potentials_from_particle_insertion
```
### Anaconda
When using the Anaconda distribution, it is safer to run the conda version of pip as follows:
```
conda install pip
conda install git
pip install git+https://github.com/MaartenBransen/potentials_from_particle_insertion
```
### Updating
To update to the most recent version, use `pip install` with the `--upgrade` flag set:
```
pip install --upgrade git+https://github.com/MaartenBransen/potentials_from_particle_insertion
```

## Usage
After installing it can be imported like any normal package by writing e.g. 
```Python
import potentials_from_particle_insertion
```
to import everything, or something like
```Python
from potentials_from_particle_insertion import rdf_dist_hist_3d, run_iteration
```
to only import specific functions.

For a complete API reference see [the documentation](https://maartenbransen.github.io/potentials_from_particle_insertion/) (may be private) or download `index.html` from the `gh-pages` branch and open it with your browser. There are two example usage scripts in `/examples/` for solving for the pair-potential of a randomly generated ideal gas in 2D and 3D in periodic and nonperiodic boundary conditions respectively.
