"""
-------------------------------------------------------------------------------

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

-------------------------------------------------------------------------------
"""

__version__ = '0.6.0'

from .pairpotential_iterator import (
    run_iteration,
    run_iterator_fitfunction,
    #run_iterator_fitfunction2,
    #run_iterator_fitfunction3,
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
    'run_iteration',
    'run_iterator_fitfunction',
    'run_iterator_fitfunction2',
    'run_iterator_fitfunction3',
    'rdf_dist_hist_2d',
    'rdf_dist_hist_3d',
    'rdf_insertion_binned_2d',
    'rdf_insertion_binned_3d',
    'rdf_insertion_exact_3d',
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