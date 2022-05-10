"""
Maarten Bransen, 2020
m.bransen@uu.nl
"""

__version__ = '0.5.0'

from .pairpotential_iterator import run_iteration
from .distribution_functions import rdf_insertion_binned_3d,rdf_insertion_exact_3d,\
    rdf_dist_hist_3d,rdf_insertion_binned_2d,rdf_dist_hist_2d

#define all for doing `from .. import *`
__all__ = [
    'run_iteration',
    'rdf_insertion_binned_2d',
    'rdf_insertion_binned_3d',
    'rdf_insertion_exact_3d',
    'rdf_dist_hist_2d',
    'rdf_dist_hist_3d',
]

#add submodules to pdoc ignore list for generated documentation
__pdoc__ = {
    'pairpotential_iterator' : False,
    'distribution_functions' : False,
    'geometry' : False,
    'generate_coordinates' : False
}