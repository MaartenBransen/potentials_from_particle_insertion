"""
Maarten Bransen, 2020
m.bransen@uu.nl
"""

__version__ = '0.3'

from .pairpotential_iterator import run_iteration,run_iterator_fitfunction,run_iterator_fitfunction2
from .distribution_functions import rdf_insertion_binned_3d,rdf_insertion_exact_3d,\
    rdf_dist_hist_3d,rdf_insertion_binned_2d,rdf_dist_hist_2d,rdf_dist_hist_2d_circularboundary

#define all for doing `from .. import *`
__all__ = [
    'run_iteration',
    'run_iterator_fitfunction',
    'run_iterator_fitfunction2'
    'rdf_insertion_binned_2d',
    'rdf_insertion_binned_3d',
    'rdf_insertion_exact_3d',
    'rdf_dist_hist_2d',
    'rdf_dist_hist_2d_circularboundary',
    'rdf_dist_hist_3d',
]

#add submodules to pdoc ignore list for generated documentation
__pdoc__ = {
    'pairpotential_iterator' : False,
    'distribution_functions' : False,
    'geometry' : False,
    'generate_coordinates' : False
}