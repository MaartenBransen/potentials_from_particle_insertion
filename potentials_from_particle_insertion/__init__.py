"""
Description
-----------
Python package with a set of functions for calculating the g(r) using the 
Widom test-particle insertion method, and/or for fitting the g(r) to solve for
the pair-potential using particle coordinates from e.g. confocal microscopy or 
cryo-TEM.

The codes for this package were developed as part of the work in reference [1],
where the iterative use of test-particle insertion to solve for the pair 
potential was based on [2] with correction for periodic or finite boundary 
conditions based on equations described in [3,4]

References
----------
[1] Bransen, M. (2024). Measuring interactions between colloidal 
(nano)particles. PhD thesis, Utrecht University.

[2] Stones, A. E., Dullens, R. P. A., & Aarts, D. G. A. L. (2019). Model-Free 
Measurement of the Pair Potential in Colloidal Fluids Using Optical Microscopy.
Physical Review Letters, 123(9), 098002. 
https://doi.org/10.1103/PhysRevLett.123.098002

[3] Markus Seserno (2014). How to calculate a three-dimensional g(r) under
periodic boundary conditions.
https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf

[4] Kopera, B. A. F., & Retsch, M. (2018). Computing the 3D Radial Distribution
Function from Particle Positions: An Advanced Analytic Approach. Analytical 
Chemistry, 90(23), 13909–13914.
https://doi.org/10.1021/acs.analchem.8b03157

License
-------
MIT license

Copyright (c) 2024 Maarten Bransen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = '1.0.0'

from .pairpotential_iterator import (
    run_iteration,
    run_iterator_fitfunction,
)

from .distribution_functions import (
    rdf_dist_hist_2d,
    rdf_dist_hist_3d,
    rdf_insertion_binned_2d,
    rdf_insertion_binned_3d,
    rdf_insertion_exact_3d,
)

from .utility import (
    save_TPI_results,
    load_TPI_results,
)


#define all for doing `from .. import *`
__all__ = [
    'rdf_dist_hist_2d',
    'rdf_dist_hist_3d',
    'rdf_insertion_binned_2d',
    'rdf_insertion_binned_3d',
    'rdf_insertion_exact_3d',
    'run_iteration',
    'run_iterator_fitfunction',
    'save_TPI_results',
    'load_TPI_results',
]

#add submodules to pdoc ignore list for generated documentation
__pdoc__ = {
    'pairpotential_iterator' : False,
    'distribution_functions' : False,
    'geometry' : False,
    'generate_coordinates' : False,
    'utility' : False
}