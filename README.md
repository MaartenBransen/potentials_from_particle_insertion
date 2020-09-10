# potentials_from_particle_insertion
## Overview
Python package with a set of functions for calculating the g(r) using the Widom test-particle insertion method, and/or for fitting the g(r) to solve for the pair-potential using particle coordinates from e.g. confocal microscopy

## Installation
Download/clone the repository and place the `/potentials_from_particle_insertion` folder in your working directory or the default package directory (`../anaconda3/Lib/site-packages/` when using the Anaconda distibution). It can then be imported like any normal package by writing e.g. 
```Python
import potentials_from_particle_insertion
```
to import everything, or something like
```Python
from potentials_from_particle_insertion import rdf_dist_hist_3d, run_iteration
```
to only import specific functions.

## Usage
For a complete API reference see [the documentation](https://maartenbransen.github.io/potentials_from_particle_insertion/) (may be private) or download `/docs/index.html` and open it with your browser. There are two example usage scripts in `/examples/` for solving for the pair-potential of a randomly generated ideal gas in 2D and 3D in periodic and nonperiodic boundary conditions respectively.
