# potentials_from_particle_insertion
## Overview
Python package with a set of functions for calculating the g(r) using the Widom test-particle insertion method, and/or for fitting the g(r) to solve for the pair-potential using particle coordinates from e.g. confocal microscopy or cryo-TEM.

The codes for this package were developed as part of the work in reference [1], where the iterative use of test-particle insertion to solve for the pair potential was based on [2] with correction for periodic or finite boundary conditions based on equations described in [3,4].

### References
[1] Bransen, M. (2024). Measuring interactions between colloidal (nano)particles. PhD thesis, Utrecht University.

[2] Stones, A. E., Dullens, R. P. A., & Aarts, D. G. A. L. (2019). [Model-Free Measurement of the Pair Potential in Colloidal Fluids Using Optical Microscopy](https://doi.org/10.1103/PhysRevLett.123.098002). Physical Review Letters, 123(9), 098002. 

[3] Markus Seserno (2014). [How to calculate a three-dimensional g(r) under periodic boundary conditions](https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf).

[4] Kopera, B. A. F., & Retsch, M. (2018). [Computing the 3D Radial Distribution Function from Particle Positions: An Advanced Analytic Approach](https://doi.org/10.1021/acs.analchem.8b03157). Analytical Chemistry, 90(23), 13909–13914.

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

For a complete API reference see [the documentation](https://maartenbransen.github.io/potentials_from_particle_insertion/) or download `index.html` from the `gh-pages` branch and open it with your browser. There are example usage scripts in `/examples/` for solving for the pair-potential of a randomly generated ideal gas in 2D and 3D in periodic and nonperiodic boundary conditions respectively, as well as examples for multi-component systems and using fit functions.
