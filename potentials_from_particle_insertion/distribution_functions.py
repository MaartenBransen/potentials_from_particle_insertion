"""
Set of functions for calculating the radial distribution function using test-
particle insertion with a known pair potential.

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
#%% imports
#external imports
import numpy as np
from scipy.spatial import cKDTree
from warnings import warn

#check numba available
try:
    import numba as nb
    _numba_available = True
except ImportError:
    warn("numba not detected, falling back to pure python. Install numba for "
         "a performance increase.")
    _numba_available = False

#internal imports
from .generate_coordinates import (
    _rand_coord_in_box,
    _rand_coord_in_circle,
    _rand_coord_in_sphere,
    _rand_coord_at_dist,
    _rand_coord_on_sphere,
    _coord_grid_in_box,
)

from .geometry import (
    _sphere_shell_vol_frac_in_cuboid,
    _sphere_shell_vol_frac_in_cuboid_periodic,
    _sphere_shell_vol_frac_in_sphere,
    _circle_ring_area_frac_in_rectangle,
    _circle_ring_area_frac_in_rectangle_periodic,
    _circle_ring_area_frac_in_circle,
)


#%% public definitions

def rdf_dist_hist_2d(coordinates,rmin=0,rmax=10,dr=None,
    handle_edge='rectangle',boundary=None,density=None,quiet=False,
    neighbors_upper_bound=None,workers=1):
    """calculates g(r) via a 'conventional' distance histogram method for a 
    set of 2D coordinate sets. Provided for convenience.

    Parameters
    ----------
    coordinates : numpy.array or list-like of numpy.array
        list of sets of coordinates, where each item along the 0th dimension is
        a N*2 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one TEM image, a time step from a 
        video, etc.), with each element of the array of form  `[y,x]`. Each 
        set of coordinates is not required to have the same number of particles
        but must have the same particle density. When no boundaries or a single
        boundary is given, it is assumed all coordinate sets share these same
        boundaries.
    rmin : float, optional
        lower bound for the pairwise distance, left edge of 0th bin. The 
        default is 0.
    rmax : float, optional
        upper bound for the pairwise distance, right edge of last bin. The 
        default is 10.
    dr : float, optional
        bin width for the pairwise distance bins. The default is (rmax-rmin)/20
    handle_edge : one of ['none','rectangle','periodic rectangle','circle', (list of) callable], optional
        specifies how to correct for edge effects in the radial distribution 
        function. These edge effects occur due to particles closer than `rmax`
        to any of the boundaries missing part of their neighbour shells in the 
        dataset. The following options for correcting for this are available:
            
        *   `'none'` or `None` or `False` (all equivalent): do not correct 
            for edge effects at all. Note that in order to calculate the 
            particle density, cuboidal boundaries are assumed even when 
            `boundary` is not specified. This can be overridden by explicitely
            giving the particle density using the `density` parameter.
        *   `'rectangle'`: correct for the missing volume in a square or 
            rectangular boundary.
        *   `'periodic rectangle'`: like `'rectangle'`, except in periodic 
            boundary conditions (i.e. one side wraps around to the other), 
            based on ref [1].
        *   `'circle'`: correct for missing volume in a spherical boundary.
        *   a custom callable function (or list thereof) can be given to 
            correct for arbitrary and/or mixed boundary conditions. This 
            function must take three arguments: a numpy array of bin edges, an
            N×2 numpy array of coordinates and a boundary as specified in 
            `boundary`, and return an N × `len(bin edges)-1` numpy array with 
            a value between 0 and 1 specifying the fraction of the volume of 
            each circular shell that is within the boundary.
        
        The default is 'rectangle'.
    
    boundary : array-like, optional
        positions of the walls that define the bounding box of the coordinates.
        The format must be given as `((ymin,ymax),(xmin,xmax))` when 
        `handle_edge` is `'none'`, `'rectangle'` or `'periodic rectangle'`. 
        When `handle_edge='circle'` the coordinates of the origin and radius of
        the bouding circle must be given as `(y,x,radius)`. For custom boundary
        handling, `boundary` can be any format as required by the edge 
        edge correction function(s) passed to `handle_edge`.
        
        If all coordinate sets share the same boundary, a single such boundary 
        can be given, otherwise a list of such array-likes of the same length 
        as coordinates can be given for specifying boundaries of each set in 
        `coordinates` separately. The default `None` which determines the 
        smallest set of boundaries encompassing a set of coordinates.
    density : float or callable or list of callable, optional
        number density of particles in the box to use for normalizing the 
        values. The default is the average density based on `coordinates` and
        `boundary`. When using custom edge correction in `handle_edge` it is
        possible to instead pass a (list of) callable that takes the number of
        particles and boundary and returns the number density.
    quiet : bool, optional
        if True, no output is printed to the terminal by this function call. 
        The default is False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for reducing memory consumption in datasets with 
        dimensions much larger than rmax. The default is None, which takes the
        total number of particles in the coordinate set as upper limit.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. Note: this is
        ignored when `handle_edge='periodic rectangle'`. The default is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    bincounts : numpy.array
        values for the bins of the radial distribution function
    
    References
    ----------
    [1] Markus Seserno (2014). How to calculate a three-dimensional g(r) under
    periodic boundary conditions.
    https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf
    """
    #check the type of edge handling, and pass off to appropriate subfunction:
    #don't correct for edge effects, but assume rectangular box otherwise
    if handle_edge is None or handle_edge is False or handle_edge == 'none':
        return _rdf_dist_hist_2d_rectangle(coordinates,rmin=rmin,rmax=rmax,
            dr=dr,boundary=boundary,density=density,periodic_boundary=False,
            handle_edge=False,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges in rectangular or square box
    elif handle_edge == 'rectangle':
        return _rdf_dist_hist_2d_rectangle(coordinates,rmin=rmin,rmax=rmax,
            dr=dr,boundary=boundary,density=density,periodic_boundary=False,
            handle_edge=True,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges in periodic boundary conditions in rectangle/square box
    elif handle_edge == 'periodic rectangle':
        return _rdf_dist_hist_2d_rectangle(coordinates,rmin=rmin,rmax=rmax,
            dr=dr,boundary=boundary,density=density,periodic_boundary=True,
            handle_edge=True,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges in circular box
    elif handle_edge == 'circle':
        return _rdf_dist_hist_2d_circle(coordinates,rmin=rmin,rmax=rmax,dr=dr,
            boundary=boundary,density=density,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges using arbitrary correction funcs
    elif callable(handle_edge) or \
        (type(handle_edge)==list and callable(handle_edge[0])):
        return _rdf_dist_hist_2d_custom(coordinates,handle_edge,boundary,
            density,rmin=rmin,rmax=rmax,dr=dr,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #error for unrecognised options for handle_edge
    else:
        raise ValueError(f'{handle_edge} not a valid option for `handle_edge`')

def rdf_dist_hist_3d(coordinates,rmin=0,rmax=10,dr=None,handle_edge='cuboid',
        boundary=None,density=None,quiet=False,neighbors_upper_bound=None,
        workers=1):
    """calculates g(r) via a 'conventional' distance histogram method for a 
    set of 3D coordinate sets. Provided for convenience.

    Parameters
    ----------
    coordinates : numpy.array or list-like of numpy.array
        list of sets of coordinates, where each item along the 0th dimension is
        a N*3 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one z-stack, a time step from a 
        video, etc.), with each element of the array of form  `[z,y,x]`. Each 
        set of coordinates is not required to have the same number of particles
        but must have the same particle density. When no boundaries or a single
        boundary is given, it is assumed all coordinate sets share these same
        boundaries.
    rmin : float, optional
        lower bound for the pairwise distance, left edge of 0th bin. The 
        default is 0.
    rmax : float, optional
        upper bound for the pairwise distance, right edge of last bin. The 
        default is 10.
    dr : float, optional
        bin width for the pairwise distance bins. The default is (rmax-rmin)/20
    handle_edge : one of ['none','cuboid','periodic cuboid','sphere', (list of) callable], optional
        specifies how to correct for edge effects in the radial distribution 
        function. These edge effects occur due to particles closer than `rmax`
        to any of the boundaries missing part of their neighbour shells in the 
        dataset. The following options for correcting for this are available:
            
        *   `'none'` or `None` or `False` (all equivalent): do not correct 
            for edge effects at all. Note that in order to calculate the 
            particle density, cuboidal boundaries are assumed even when 
            `boundary` is not specified. This can be overridden by explicitely
            giving the particle density using `density`.
        *   `'cuboid'`: correct for the missing volume in cuboidal boundary
            conditions, e.g. a 3D rectangular box with right angles. Based on
            ref. [1]
        *   `'periodic cuboid'`: like `'cuboid'`, except in periodic 
            boundary conditions (i.e. one side wraps around to the other). 
            Based on ref. [2].
        *   `'sphere'`: correct for missing volume in spherical boundary
            conditions
        *   a custom callable function (or list thereof) can be given to 
            correct for arbitrary and/or mixed boundary conditions. This 
            function must take three arguments: a numpy array of bin edges, an
            N×3 numpy array of coordinates and a boundary as specified in 
            `boundary`, and return an N × `len(bin edges)-1` numpy array with 
            a value between 0 and 1 specifying the fraction of the volume of 
            each spherical shell that is within the boundary.
        
        The default is 'cuboid'.
    
    boundary : array-like, optional
        positions of the walls that define the bounding box of the coordinates.
        The format must be given as `((zmin,zmax),(ymin,ymax),(xmin,xmax))` 
        when `handle_edge` is `'none'`, `'cuboid'` or `'periodic cuboid'`. When
        `handle_edge='sphere'` the coordinates of the origin and radius of the
        bouding sphere must be given as `(z,y,x,radius)`.
        
        If all coordinate sets share the same boundary a single such boundary 
        can be given, otherwise a list of such array-likes of the same length 
        as coordinates can be given for specifying boundaries of each set in 
        `coordinates` separately. The default is `None` which determines the 
        smallest set of boundaries encompassing a set of coordinates.
    density : float or callable or list of callable, optional
        number density of particles in the box to use for normalizing the 
        values. The default is the average density based on `coordinates` and
        `boundary`. When using custom edge correction in `handle_edge` it is
        possible to instead pass a (list of) callable that takes the number of
        particles and boundary and returns the number density.
    quiet : bool, optional
        if True, no output is printed to the terminal by this function call. 
        The default is False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for reducing memory consumption in datasets with 
        dimensions much larger than rmax. The default is None, which takes the
        total number of particles in the coordinate set as upper limit.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. Note: this is
        ignored when `handle_edge='periodic cuboid'`. The default is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    bincounts : numpy.array
        values for the bins of the radial distribution function
    
    References
    ----------
    [1] Kopera, B. A. F., & Retsch, M. (2018). Computing the 3D Radial 
    Distribution Function from Particle Positions: An Advanced Analytic 
    Approach. Analytical Chemistry, 90(23), 13909–13914. 
    https://doi.org/10.1021/acs.analchem.8b03157
    
    [2] Markus Seserno (2014). How to calculate a three-dimensional g(r) under
    periodic boundary conditions.
    https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf
    """
    #check the type of edge handling, and pass off to appropriate subfunction:
    #don't correct for edge effects, but assume cuboidal box otherwise
    if handle_edge is None or handle_edge is False or handle_edge == 'none':
        return _rdf_dist_hist_3d_cuboid(coordinates,rmin=rmin,rmax=rmax,dr=dr,
            boundary=boundary,density=density,periodic_boundary=False,
            handle_edge=False,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges in cuboidal box
    elif handle_edge == 'cuboid':
        return _rdf_dist_hist_3d_cuboid(coordinates,rmin=rmin,rmax=rmax,dr=dr,
            boundary=boundary,density=density,periodic_boundary=False,
            handle_edge=True,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges in periodic boundary conditions (cube or cuboid)
    elif handle_edge == 'periodic cuboid':
        return _rdf_dist_hist_3d_cuboid(coordinates,rmin=rmin,rmax=rmax,dr=dr,
            boundary=boundary,density=density,periodic_boundary=True,
            handle_edge=True,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges in spherical box
    elif handle_edge == 'sphere':
        return _rdf_dist_hist_3d_sphere(coordinates,rmin=rmin,rmax=rmax,dr=dr,
            boundary=boundary,density=density,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges using arbitrary correction funcs
    elif callable(handle_edge) or \
        (type(handle_edge)==list and callable(handle_edge[0])):
        return _rdf_dist_hist_3d_custom(coordinates,handle_edge,boundary,
            density,rmin=rmin,rmax=rmax,dr=dr,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #error for unrecognised options for handle_edge
    else:
        raise ValueError(f'{handle_edge} not a valid option for `handle_edge`')

def rdf_insertion_binned_2d(coordinates,pairpotential,rmin=0,rmax=10,dr=None,
        handle_edge='rectangle',boundary=None,pairpotential_binedges=None,
        n_ins=1000,interpolate=True,avoid_boundary=False,
        avoid_coordinates=False,neighbors_upper_bound=None,workers=1,
        testparticle_func=None,**kwargs):
    """
    Calculate g(r) from insertion of test-particles into sets of existing
    2D coordinates, averaged over bins of width dr, and based on the pairwise 
    interaction potential u(r) (in units of kT).

    Parameters
    ----------
    coordinates : numpy.array or list-like of numpy.array
        list of sets of coordinates, where each item along the 0th dimension is
        a N*2 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one TEM image, a time step from a 
        video, etc.), with each element of the array of form  `[y,x]`. Each 
        set of coordinates is not required to have the same number of particles
        but must have the same particle density. When no boundaries or a single
        boundary is given, it is assumed all coordinate sets share these same
        boundaries.
    pairpotential : iterable
        list of values for the pairwise interaction potential. Must have length
        of `len(pairpotential_binedges)-1` and be in units of thermal energy kT
    rmin : float, optional
        lower bound for the pairwise distance, left edge of 0th bin. The 
        default is 0.
    rmax : float, optional
        upper bound for the pairwise distance, right edge of last bin. The 
        default is 10.
    dr : float, optional
        bin width for the pairwise distance bins. The default is (rmax-rmin)/20
    handle_edge : one of ['none','rectangle','periodic rectangle','circle', (list of) callable], optional
        specifies how to correct for edge effects in the radial distribution 
        function. These edge effects occur due to particles closer than `rmax`
        to any of the boundaries missing part of their neighbour shells in the 
        dataset. The following options for correcting for this are available:
            
        *   `'rectangle'`: correct for the missing volume in a square or 
            rectangular boundary.
        *   `'periodic' rectangle'`: like `'rectangle'`, except in periodic 
            boundary conditions (i.e. one side wraps around to the other), 
            based on ref [1].
        *   `'circle'`: correct for missing volume in a spherical boundary.
        *   a custom callable function (or list thereof) can be given to 
            correct for arbitrary and/or mixed boundary conditions. This 
            function must take three arguments: a numpy array of bin edges, an
            N×2 numpy array of coordinates and a boundary as specified in 
            `boundary`, and return an N × `len(bin edges)-1` numpy array with 
            a value between 0 and 1 specifying the fraction of the volume of 
            each spherical shell that is within the boundary.
        
        The default is 'rectangle'.
    boundary : array-like or list of such, optional
        positions of the walls that define the bounding box of the coordinates.
        The format must be given as `((ymin,ymax),(xmin,xmax))` when 
        `handle_edge` is ``'rectangle'` or `'periodic rectangle'`. 
        When `handle_edge='circle'` the coordinates of the origin and radius of
        the bouding circle must be given as `(y,x,radius)`. For custom boundary
        handling, `boundary` can be any format as required by the edge 
        edge correction function(s) passed to `handle_edge`.
        
        If all coordinate sets share the same boundary, a single such boundary 
        can be given, otherwise a list of such array-likes of the same length 
        as coordinates can be given for specifying boundaries of each set in 
        `coordinates` separately. The default is `None` which determines the 
        smallest set of boundaries encompassing a set of coordinates.
    pairpotential_binedges : iterable, optional
        bin edges corresponding to the values in `pairpotential`. The default 
        is None, which uses the bins defined by `rmin`, `rmax` and `dr`.
    n_ins : int, optional
        the number of test-particles to insert into each item in `coordinates`.
        The default is 1000.
    interpolate : bool, optional
        whether to use linear interpolation for calculating the interaction of 
        two particles using the values in `pairpotential`. The default is True.
        If False, the nearest bin value is used.
    avoid_boundary : bool, optional
        if True, all test-particles are inserted at least `rmax` away from any 
        of the surfaces defined in `boundary` to avoid effects of the finite
        volume of the bounding box. The default is False, which uses an 
        analytical correction factor for missing volume of test-particles near 
        the boundaries. This parameter is ignored when using custom edge
        handling.
    avoid_coordinates : bool, optional
        whether to insert test-particles at least `rmin` away from the center 
        of any of the 'real' coordinates in `coordinates`. The default is 
        False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for reducing memory consumption in datasets with 
        dimensions much larger than rmax. The default is None, which takes the
        total number of particles in the coordinate set as upper limit.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. Note: this is
        ignored when `handle_edge='periodic rectangle'`. The default is 1.
    testparticle_func : callable or list thereof
        function that generates test-particle coordinates in the bounding box, 
        required when using custom edge handling (and ignored otherwise). When
        `avoid_coordinates=True` this function must take 4 arguments: a 
        boundary as given in `boundary`, a N×2 numpy.array of coordinates to
        avoid, a float specifing the radius around the coordinates to avoid, 
        and an integer giving the number of coordinates to generate. Otherwise,
        it must only take the boundary and the number of coordinates to 
        generate as arguments. It must return an `n_ins` × 2 numpy array of 
        coordinates.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    pair_correlation : numpy.array
        values for the rdf / pair correlation function in each bin
    counter : numpy.array
        number of pair counts that contributed to the (mean) values in each bin
    
    References
    ----------
    [1] Markus Seserno (2014). How to calculate a three-dimensional g(r) under
    periodic boundary conditions.
    https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf
    """
    #check the type of edge handling, and pass off to appropriate subfunction:
    #correct for edges in rectangular or square box
    if handle_edge == 'rectangle':
        return _rdf_insertion_binned_2d_rectangle(coordinates, pairpotential, 
            periodic_boundary=False,rmin=rmin,rmax=rmax,
            dr=dr,boundary=boundary,pairpotential_binedges=pairpotential_binedges,
            n_ins=n_ins,interpolate=interpolate,avoid_boundary=avoid_boundary,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers,
            **kwargs)
    
    #correct for edges in periodic boundary conditions in rectangle/square box
    elif handle_edge == 'periodic rectangle':
        return _rdf_insertion_binned_2d_rectangle(coordinates, pairpotential, 
            periodic_boundary=True,rmin=rmin,rmax=rmax,dr=dr,
            boundary=boundary,pairpotential_binedges=pairpotential_binedges,
            n_ins=n_ins,interpolate=interpolate,avoid_boundary=avoid_boundary,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers,
            **kwargs)
    
    #correct for edges in circular box
    elif handle_edge == 'circle':
        return _rdf_insertion_binned_2d_circle(coordinates, pairpotential, 
            rmin=rmin,rmax=rmax,dr=dr,boundary=boundary,
            pairpotential_binedges=pairpotential_binedges,n_ins=n_ins,
            interpolate=interpolate,avoid_boundary=avoid_boundary,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers,
            **kwargs)
    
    #correct for edges using arbitrary correction funcs
    elif callable(handle_edge) or \
        (type(handle_edge)==list and callable(handle_edge[0])):
        return _rdf_insertion_binned_2d_custom(coordinates, pairpotential, 
            handle_edge,boundary,testparticle_func,rmin=rmin,rmax=rmax,dr=dr,
            pairpotential_binedges=pairpotential_binedges,n_ins=n_ins,
            interpolate=interpolate,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers,
            **kwargs)
    
    #error for unrecognised options for handle_edge
    else:
        raise ValueError(f'{handle_edge} not a valid option for `handle_edge`')

def rdf_insertion_binned_3d(coordinates,pairpotential,rmin=0,rmax=10,dr=None,
        handle_edge='cuboid',boundary=None,pairpotential_binedges=None,
        n_ins=1000,interpolate=True,avoid_boundary=False,
        avoid_coordinates=False,neighbors_upper_bound=None,workers=1,
        testparticle_func=None,**kwargs):
    """Calculate g(r) from insertion of test-particles into sets of existing
    2D coordinates, averaged over bins of width dr, and based on the pairwise 
    interaction potential u(r) (in units of kT).

    Parameters
    ----------
    coordinates : numpy.array or list-like of numpy.array
        list of sets of coordinates, where each item along the 0th dimension is
        a N*3 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one z-stack, a time step from a 
        video, etc.), with each element of the array of form  `[z,y,x]`. Each 
        set of coordinates is not required to have the same number of particles
        but must have the same particle density. When no boundaries or a single
        boundary is given, it is assumed all coordinate sets share these same
        boundaries.
    pairpotential : iterable
        list of values for the pairwise interaction potential. Must have length
        of `len(pairpotential_binedges)-1` and be in units of thermal energy kT
    rmin : float, optional
        lower bound for the pairwise distance, left edge of 0th bin. The 
        default is 0.
    rmax : float, optional
        upper bound for the pairwise distance, right edge of last bin. The 
        default is 10.
    dr : float, optional
        bin width for the pairwise distance bins. The default is (rmax-rmin)/20
    handle_edge : one of ['none','cuboid','periodic cuboid','sphere', (list of) callable], optional
        specifies how to correct for edge effects in the radial distribution 
        function. These edge effects occur due to particles closer than `rmax`
        to any of the boundaries missing part of their neighbour shells in the 
        dataset. The following options for correcting for this are available:
            
        *   `'none'` or `None` or `False` (all equivalent): do not correct 
            for edge effects at all. Note that in order to calculate the 
            particle density, cuboidal boundaries are assumed even when 
            `boundary` is not specified. This can be overridden by explicitely
            giving the particle density using `density`.
        *   `'cuboid'`: correct for the missing volume in cuboidal boundary
            conditions, e.g. a 3D rectangular box with right angles. Based on
            ref. [1]
        *   `'periodic' cuboid'`: like `'cuboid'`, except in periodic 
            boundary conditions (i.e. one side wraps around to the other). 
            Based on ref. [2].
        *   `'sphere'`: correct for missing volume in spherical boundary
            conditions
        *   a custom callable function (or list thereof) can be given to 
            correct for arbitrary and/or mixed boundary conditions. This 
            function must take three arguments: a numpy array of bin edges, an
            N×3 numpy array of coordinates and a boundary as specified in 
            `boundary`, and return an N × `len(bin edges)-1` numpy array with 
            a value between 0 and 1 specifying the fraction of the volume of 
            each spherical shell that is within the boundary.
        
        The default is 'cuboid'.
    
    boundary : array-like, optional
        positions of the walls that define the bounding box of the coordinates.
        The format must be given as `((zmin,zmax),(ymin,ymax),(xmin,xmax))` 
        when `handle_edge` is `'none'`, `'cuboid'` or `'periodic cuboid'`. When
        `handle_edge='sphere'` the coordinates of the origin and radius of the
        bouding sphere must be given as `(z,y,x,radius)`.
        
        If all coordinate sets share the same boundary a single such boundary 
        can be given, otherwise a list of such array-likes of the same length 
        as coordinates can be given for specifying boundaries of each set in 
        `coordinates` separately. The default is `None` which determines the 
        smallest set of boundaries encompassing a set of coordinates.
    pairpotential_binedges : iterable, optional
        bin edges corresponding to the values in `pairpotential`. The default 
        is None, which uses the bins defined by `rmin`, `rmax` and `dr`.
    n_ins : int, optional
        the number of test-particles to insert into each item in `coordinates`.
        The default is 1000.
    interpolate : bool, optional
        whether to use linear interpolation for calculating the interaction of 
        two particles using the values in `pairpotential`. The default is True.
        If False, the nearest bin value is used.
    avoid_boundary : bool, optional
        if True, all test-particles are inserted at least `rmax` away from any 
        of the surfaces defined in `boundary` to avoid effects of the finite
        volume of the bounding box. The default is False, which uses an 
        analytical correction factor for missing volume of test-particles near 
        the boundaries. This parameter is ignored when using custom edge
        handling.
    avoid_coordinates : bool, optional
        whether to insert test-particles at least `rmin` away from the center 
        of any of the 'real' coordinates in `coordinates`. The default is 
        False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for reducing memory consumption in datasets with 
        dimensions much larger than rmax. The default is None, which takes the
        total number of particles in the coordinate set as upper limit.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. Note: this is
        ignored when `handle_edge='periodic rectangle'`. The default is 1.
    testparticle_func : callable or list thereof
        function that generates test-particle coordinates in the bounding box, 
        required when using custom edge handling (and ignored otherwise). When
        `avoid_coordinates=True` this function must take 4 arguments: a 
        boundary as given in `boundary`, a N×3 numpy.array of coordinates to
        avoid, a float specifing the radius around the coordinates to avoid, 
        and an integer giving the number of coordinates to generate. Otherwise,
        it must only take the boundary and the number of coordinates to 
        generate as arguments. It must return an `n_ins`×3 numpy array of 
        coordinates.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    pair_correlation : numpy.array
        values for the rdf / pair correlation function in each bin
    counter : numpy.array
        number of pair counts that contributed to the (mean) values in each bin
    
    References
    ----------
    [1] Kopera, B. A. F., & Retsch, M. (2018). Computing the 3D Radial 
    Distribution Function from Particle Positions: An Advanced Analytic 
    Approach. Analytical Chemistry, 90(23), 13909–13914. 
    https://doi.org/10.1021/acs.analchem.8b03157
    
    [2] Markus Seserno (2014). How to calculate a three-dimensional g(r) under
    periodic boundary conditions.
    https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf
    """
    #check the type of edge handling, and pass off to appropriate subfunction:
    #don't correct for edge effects, but assume rectangular box otherwise
    if handle_edge is None or handle_edge is False or handle_edge == 'none':
        return _rdf_insertion_binned_2d_rectangle(coordinates, pairpotential, 
            periodic_boundary=False,rmin=rmin,rmax=rmax,
            dr=dr,boundary=boundary,
            pairpotential_binedges=pairpotential_binedges,n_ins=n_ins,
            interpolate=interpolate,avoid_boundary=avoid_boundary,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers,
            **kwargs)
    
    #correct for edges in cubic or cuboidal box
    elif handle_edge == 'cuboid':
        return _rdf_insertion_binned_3d_cuboid(coordinates, pairpotential, 
            periodic_boundary=False,rmin=rmin,rmax=rmax,
            dr=dr,boundary=boundary,
            pairpotential_binedges=pairpotential_binedges,n_ins=n_ins,
            interpolate=interpolate,avoid_boundary=avoid_boundary,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers,
            **kwargs)
    
    #correct for edges in periodic boundary conditions in cubic/cuboidal box
    elif handle_edge == 'periodic cuboid':
        return _rdf_insertion_binned_3d_cuboid(coordinates, pairpotential, 
            periodic_boundary=True,rmin=rmin,rmax=rmax,
            dr=dr,boundary=boundary,
            pairpotential_binedges=pairpotential_binedges,n_ins=n_ins,
            interpolate=interpolate,avoid_boundary=avoid_boundary,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers,
            **kwargs)
    
    #correct for edges in spherical box
    elif handle_edge == 'sphere':
        return _rdf_insertion_binned_3d_sphere(coordinates, pairpotential, 
            rmin=rmin,rmax=rmax,dr=dr,boundary=boundary,
            pairpotential_binedges=pairpotential_binedges,n_ins=n_ins,
            interpolate=interpolate,avoid_boundary=avoid_boundary,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers,
            **kwargs)
    
    #correct for edges using arbitrary correction funcs
    elif callable(handle_edge) or \
        (type(handle_edge)==list and callable(handle_edge[0])):
        return _rdf_insertion_binned_3d_custom(coordinates, pairpotential, 
            handle_edge,boundary,testparticle_func,rmin=rmin,rmax=rmax,dr=dr,
            pairpotential_binedges=pairpotential_binedges,n_ins=n_ins,
            interpolate=interpolate,avoid_boundary=avoid_boundary,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers,
            **kwargs)
    
    #error for unrecognised options for handle_edge
    else:
        raise ValueError(f'{handle_edge} not a valid option for `handle_edge`')

    
def rdf_insertion_exact_3d(coordinates,pairpotential,rmax,dr,boundary,
        pairpotential_binedges=None,gen_prob_reps=1000,shell_prob_reps=10,
        interpolate=True,use_numba=True,rmin=0):
    """
    .. warning::
       This function underwent very minimal testing and is computationally 
       inefficient. Use at your own risk.
    
    calculate g(r) from particle insertion method using particle coordinates
    and pairwise interaction potential u(r) (in units of kT). Inserts test-
    particles at a specific r for every real particle. Implementation based on
    ref [1]

    Parameters
    ----------
    coordinates : list of numpy.array of floats of shape nt*n*3
        The coordinates of particles in the system
    pairpotential : list of floats
        the value of the interparticle potential in each bin of 0 to rmax+dr 
        in steps of dr. The value is assumed to correspond to the centrepoint
        of the bins.
    rmax : float
        cut-off value for the radius. Interactions beyond this range are 
        considered negligable. Right edge of last bin in the pair potential
    dr : float
        bin width/step size of the interparticle distance used for g(r) and the
        pair potential.
    boundary : tuple of floats of form ((zmin,zmax),(ymin,ymax),(xmin,xmax))
        defines the boundaries of the box in which coordinates may exist.
    gen_prob_reps : int, optional
        number of trial particles used to evaluate the general probability of
        placing a particle at random coordinates in the box defined by
        `boundary`. The default is 1000.
    shell_prob_reps : int, optional
        number of trial particles per reference coordinate in `coordinates` 
        used to evaluate the probability of placing a particle in each distance
        bin. The default is 10.
    interpolate : bool, optional
        If true, the pair potential is linearly interpolated between bin 
        centres to calculate the interactions between all pairs of particles. 
        If false, the value from the nearest bin is taken which is
        slower to compute. The default is True.

    Returns
    -------
    pair_correlation : list of float
        the pair correlation / radial distribution functions evaluated in the 
        bins whose edges are defined by numpy.arange(0,rmax+dr,dr)
    counters : list of int
        The number of trialparticles evaluated for each distance bin

    References
    ----------
    [1] Stones, A. E., Dullens, R. P. A., & Aarts, D. G. A. L. (2018). Contact 
    values of pair distribution functions in colloidal hard disks by 
    test-particle insertion. The Journal of Chemical Physics, 148(24), 241102. 
    https://doi.org/10.1063/1.5038668

    """
    coordinates = _check_coordinate_input(coordinates)
    
    #get bin edges and centers
    rvals = _get_rvals(rmin, rmax, dr)
    rcent = (rvals[1:]+rvals[:-1])/2
    nt = len(coordinates)
    nr = len(rcent)
    
    #check if numba is present
    if use_numba:
        if not _numba_available:
            print('[WARNING] potentials_from_particle_insertion: numba not '+
                  'found, falling back on pure python')
            use_numba = False
    
    #set bin edges and centers
    if type(pairpotential_binedges) == type(None):
        pairpotential_binedges = rvals
    pairpotential_bincenter = (pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2
    
    if interpolate:#interpolate pair potential between points
        pot_fun = lambda dist: np.interp(dist,pairpotential_bincenter,
                                         pairpotential)
    else:#get pair potential from nearest bin (round r to bincenter)
        from scipy.interpolate import interp1d    
        pot_fun = interp1d(pairpotential_bincenter,pairpotential,
                           kind='nearest',bounds_error=False,
                           fill_value='extrapolate')
    
    #calculate average probability over the whole box to insert particle
    tot_prob = 0
    for coords in coordinates:
        trialparticles = _rand_coord_in_box(boundary,n=gen_prob_reps)
        for trialparticle in trialparticles:
            if use_numba:
                distances = _calc_squared_dist_numba(coords,trialparticle,rmax)
            else:
                distances = _calc_squared_dist(coords,trialparticle,rmax)
            #correction factor for edge here?
            tot_prob += np.exp(-np.sum(pot_fun(np.sqrt(distances))))
    
    #check and stop in case of NaN values
    if np.isnan(tot_prob):
        raise SystemError('NaN value detected')
            
    tot_prob /= (gen_prob_reps*nt)
    
    #loop over all spherical shells at distances of bin centers of r bins
    probs = np.zeros(nr)
    counters = np.zeros(len(rcent),dtype=np.uint16)
    for i,r in enumerate(rcent):
        print('\rg(r): {:.0f}%'.format(100*i/(nr-1)),end='')
        
        #loop over all sets of coordinates
        for coords in coordinates:
            
            #loop over all particles to act as reference particle in the origin
            for refcoords in coords:
                
                #create trialparticles at r from ref_particle
                trialcoords = _rand_coord_on_sphere(
                    npoints=shell_prob_reps,
                    radius=r,
                    origin=refcoords
                    )
                
                #loop over all trial particles around ref particle
                for trialparticle in trialcoords:
                    
                    #check if the trialparticle is within the box boundaries
                    if all([(coord>=bb[0] and coord<=bb[1]) for coord,bb in zip(trialparticle,boundary)]):
                        if use_numba:
                            distances = _calc_squared_dist_numba(coords,trialparticle,rmax)
                        else:
                            distances = _calc_squared_dist(coords,trialparticle,rmax)
    
                        probs[i] += np.exp(-np.sum(pot_fun(np.sqrt(distances))))
                        counters[i] +=1
                    
        probs[i] /= counters[i]
    
    pair_correlation = probs / tot_prob
    
    return pair_correlation,counters

#%% distance histogram g(r) functions

def _rdf_dist_hist_2d_rectangle(coordinates,rmin=0,rmax=10,dr=None,
        boundary=None,density=None,periodic_boundary=False,handle_edge=True,
        quiet=False,neighbors_upper_bound=None,workers=1):
    """not intended to be called directly, see top-level `rdf_dist_hist_2d`
    function. This function calculates the rdf in rectangular boundaries with
    or without correction for finite size effects, or while accounting for 
    periodic boundary conditions as used in simulations.

    Parameters
    ----------
    coordinates : numpy.array or list-like of numpy.array of floats
        list of sets of coordinates, where each item along the 0th dimension is
        a n*2 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one z-stack, a time step from a 
        video, etc.), with each element of the array of form  `[y,x]`.  Each 
        set of coordinates is not required to have the same number of particles
        and is assumed to share the same boundaries when no boundaries or only
        a single set of boundaries are given.
    rmin : float, optional
        lower bound for the pairwise distance, left edge of 0th bin. The 
        default is 0.
    rmax : float, optional
        upper bound for the pairwise distance, right edge of last bin. The 
        default is 10.
    dr : float, optional
        bin width for the pairwise distance bins. The default is (rmax-rmin)/20
    boundary : array-like, optional
        positions of the walls that define the bounding box of the coordinates,
        given as  `((ymin,ymax),(xmin,xmax))` if all coordinate sets share the 
        same boundary, or a list of such array-likes of the same length as 
        `coordinates` for specifying boundaries of each set in `coordinates` 
        separately. The default is the min and max values in the dataset along 
        each dimension.
    density : float, optional
        number density of particles in the box to use for normalizing the 
        values. The default is the average density based on `coordinates` and
        `boundary`.
    periodic_boundary : bool, optional
        whether periodic boundary conditions are used. The default is False.
    handle_edge : bool, optional
        whether to correct for edge effects in non-periodic boundary 
        conditions. The default is True.
    quiet : bool, optional
        if True, no output is printed to the terminal by this function call. 
        The default is False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. Note: this is
        ignored when `periodic_boundary=True`. The default is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    bincounts : numpy.array
        values for the bins of the radial distribution function
        
    """
    #create bins, check coordinates
    rvals = _get_rvals(rmin, rmax, dr)
    coordinates = _check_coordinate_input(coordinates)
    nf = len(coordinates)
    
    #set default boundary as min and max values in dataset
    if boundary is None:
        boundary = np.array([[
                [coord[:,0].min(),coord[:,0].max()],
                [coord[:,1].min(),coord[:,1].max()]
            ] for coord in coordinates])
    else:
        boundary = np.array(boundary)
        
        #assure list of coordinate sets
        if boundary.ndim==2:
            boundary = np.array([boundary]*nf)
    
    #check rmax and boundary for edge-handling in periodic boundary conditions
    if periodic_boundary:
        for bound in boundary:
            if bound[0,1]-bound[0,0] == bound[1,1]-bound[1,0]:
                boxlen = bound[0,1]-bound[0,0]
                if rmax > boxlen*np.sqrt(2)/2:
                    raise ValueError(
                        'rmax cannot be more than sqrt(2)/2 times the size of '
                        'a square bounding box when periodic_boundary=True, '
                        f'use rmax < {(bound[0,1]-bound[0,0])*np.sqrt(2)/2}'
                    )
            elif rmax > min(bound[:,1]-bound[:,0])/2:
                raise NotImplementedError(
                    'rmax larger than half the smallest box dimension when '
                    'periodic_boundary=True is only implemented for square '
                    'boundaries'
                )
    
    #check rmax and boundary for edge handling without periodic boundaries
    else:
        for bound in boundary:
            if rmax > max(bound[:,1]-bound[:,0])/2:
                raise ValueError(
                    'rmax cannot be larger than half the largest dimension in '
                    f'boundary, use rmax < {max(bound[:,1]-bound[:,0])/2}'
                )
    
    #set density to mean number density in dataset
    if not density:
        vol = np.prod(boundary[:,:,1]-boundary[:,:,0],axis=1)
        density = np.mean([len(coords)/v for v,coords in zip(vol,coordinates)])
    
    
    #loop over all sets of coordinates
    bincounts = np.zeros(len(rvals)-1)
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #print progress
        if not quiet:
            print(f'\rcalculating distance histogram g(r) {i+1} of '
                  f'{nf}',end='')
        
        
        #check if coords are within boundary
        if (coords<bound[:,0]).any() or (coords>=bound[:,1]).any():
            print()#newline
            warn('not all coordinates are within boundary in coordinate set '
                 f'{i}, ignoring spurious coords',RuntimeWarning)
            coords = coords[
                np.logical_and(
                    coords>=bound[:,0],
                    coords<bound[:,1]
                ).all(axis=1)
            ]
        
        #in case of periodic boundary conditions
        if periodic_boundary:
            #set up KDTree for fast neighbour finding
            #shift box boundary corner to origin for periodic KDTree
            tree = cKDTree(coords-bound[:,0],boxsize=bound[:,1]-bound[:,0])
            
            #count number of particle pairs per bin directly
            counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
            
            #optionally apply edge correction for distances beyond boxsize/2
            if handle_edge:
                
                boundarycorr = _circle_ring_area_frac_in_rectangle_periodic(
                    rvals,
                    bound[0,1]-bound[0,0]
                )
                counts = counts/boundarycorr
        
        #in nonperiodic boundary coditions
        else:
            
            #set up and query tree for fast neighbor finding
            tree = cKDTree(coords)
            
            if handle_edge:
                
                #set default neighbor_upper_bound
                if type(neighbors_upper_bound)==type(None):
                    k = len(coords)
                else:
                    k = min([neighbors_upper_bound,len(coords)])
                
                #query particles individually for per-particle edge handling
                dist,indices = tree.query(coords,k=k,distance_upper_bound=rmax,
                                          workers=workers)
                
                #remove pairs with self, padded (infinite) values and anything
                #below rmin
                dist = dist[:,1:]
                mask = np.isfinite(dist) & (dist>=rmin)
                
                #histogram the distances per reference particle and apply 
                #correction factor for missing volume to each particle (each 
                #row) and each distance bin separately
                if _numba_available:
                    counts = _apply_hist_nb(dist,mask,rvals)
                else:
                    counts = np.zeros((len(dist),len(rvals)-1))
                    for j,(row,msk) in enumerate(zip(dist,mask)):
                        counts[j] = np.histogram(row[msk],bins=rvals)[0]
                boundarycorr=_circle_ring_area_frac_in_rectangle(
                    rvals,
                    bound-coords[:,:,np.newaxis]
                    )
                counts = np.sum(counts/boundarycorr,axis=0)
                
            #otherwise find and bin pairs directly
            else:
                counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
        
        #normalize and add to overall list
        bincounts += counts / (np.pi*(rvals[1:]**2 - rvals[:-1]**2)) / \
                         (density*len(coords))
    
    #newline
    if not quiet:
        print()
    
    #average all datasets
    bincounts /= nf
    
    return rvals,bincounts

def _rdf_dist_hist_2d_circle(coordinates,rmin=0,rmax=10,dr=None,boundary=None,
        density=None,quiet=False,neighbors_upper_bound=None,workers=1):
    """not intended to be called directly, see top-level `rdf_dist_hist_2d`
    function. This function calculates the rdf in circular boundary conditions
    while correcting for edge effects.

    Parameters
    ----------
    coordinates : numpy.array or list-like of numpy.array of floats
        list of sets of coordinates, where each item along the 0th dimension is
        a n*3 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one z-stack, a time step from a 
        video, etc.), with each element of the array of form  `[y,x]`.  Each 
        set of coordinates is not required to have the same number of particles
        and is assumed to share the same boundaries when no boundaries or only
        a single set of boundaries are given.
    rmin : float, optional
        lower bound for the pairwise distance, left edge of 0th bin. The 
        default is 0.
    rmax : float, optional
        upper bound for the pairwise distance, right edge of last bin. The 
        default is 10.
    dr : float, optional
        bin width for the pairwise distance bins. The default is (rmax-rmin)/20
    boundary : tuple of floats
        coordinates of the origin and radius defining the bounding circle as
        `(y,x,r)` tuple or array-like, or a list thereof matching the length
        of  `coordinates` to specify different boundaries for each coordinate
        set.
    density : float, optional
        number density of particles in the box to use for normalizing the 
        values. The default is the average density based on `coordinates` and
        `boundary`.
    quiet : bool, optional
        if True, no output is printed to the terminal by this function call. 
        The default is False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. The default 
        is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    bincounts : numpy.array
        values for the bins of the radial distribution function
    """
    #create bins, check coordinates
    rvals = np.arange(rmin,rmax+dr,dr)
    coordinates = _check_coordinate_input(coordinates)
    nf = len(coordinates)
    
    #set default boundary or make sure the length is correct
    if boundary is None:
        boundary = []
        for coord in coordinates:
            bound = np.mean(coord,axis=0)
            rad = np.sqrt(np.sum((coord-bound)**2,axis=1)).max()
            boundary.append(np.concatenate([bound,rad]))
        boundary = np.array(boundary)
    else:
        boundary = np.array(boundary)
        if boundary.ndim == 1:
            boundary = np.broadcast_to(boundary, (nf,3))

    if len(boundary) != nf:
                raise ValueError('lengths of coordinates and boundaries must '
                                 'match')
    if any(rmax >= 2*boundary[:,2]):
        raise ValueError('rmax cannot be larger than 2 times the bounding '
                         f'circle radius, use `rmax<{2*boundary[:,2]}`')
    
    #set density to mean number density in dataset
    if not density:
        vol = np.pi*boundary[:,2]**2
        density = np.mean([len(coords)/v for v,coords in zip(vol,coordinates)])
    
    #loop over all sets of coordinates
    bincounts = np.zeros(len(rvals)-1)
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #print progress
        if not quiet:
            print(f'\rcalculating distance histogram g(r) {i+1} of {nf}',
                  end='')
         
        #check if coords are within boundary or raise warning and remove 
        d = np.sqrt(np.sum((coords - bound[:2])**2,axis=1))
        if any(d>bound[2]):
            print()#newline
            warn('not all coordinates are within boundary in coordinate set '
                 f'{i}, ignoring spurious coords',RuntimeWarning)
            coords = coords[d<=bound[2]]
            d = d[d<=bound[2]]    
        
        #set default neighbor_upper_bound
        if type(neighbors_upper_bound)==type(None):
            k = len(coords)
        else:
            k = min([neighbors_upper_bound,len(coords)])
         
        #set up and query tree for fast neighbor finding
        tree = cKDTree(coords)
        dist,indices = tree.query(coords,k=k,distance_upper_bound=rmax,
                                  workers=workers)
        
        #remove pairs with self, padded (infinite) values and any <rmin
        dist = dist[:,1:]
        mask = np.isfinite(dist) & (dist>=rmin)
        
        #histogram the distances per reference particle and apply correction 
        #factor for missing volume to each particle (each row) and each
        #distance bin separately
        if _numba_available:
            counts = _apply_hist_nb(dist,mask,rvals)
        else:
            counts = np.zeros((len(dist),len(rvals)-1))
            for j,(row,msk) in enumerate(zip(dist,mask)):
                counts[j] = np.histogram(row[msk],bins=rvals)[0]
        boundarycorr=_circle_ring_area_frac_in_circle(
            rvals,
            d,
            bound[2]
        )
        counts = np.sum(counts/boundarycorr,axis=0)
        
        #normalize and add to overall list
        bincounts += \
            counts / (np.pi*(rvals[1:]**2-rvals[:-1]**2)*density*len(coords))
    
    #newline
    if not quiet:
        print()
    
    #average all datasets
    bincounts /= nf
    
    return rvals,bincounts

def _rdf_dist_hist_2d_custom(coordinates,boundaryfunc,boundary,density,rmin=0,
        rmax=10,dr=None,quiet=False,neighbors_upper_bound=None,workers=1):
    """not intended to be called directly, see top-level `rdf_dist_hist_2d`
    function. This function calculates the rdf using boundaries and corrections
    supplied via arguments so that custom boundary geometries can be corrected
    for, as well as for a list of datasets with different boundary coditions
    between them.
    
    Parameters
    ----------
    coordinates : numpy.array or list-like of numpy.array of floats
        list of sets of coordinates, where each item along the 0th dimension is
        a n*3 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one z-stack, a time step from a 
        video, etc.), with each element of the array of form  `[y,x]`.  Each 
        set of coordinates is not required to have the same number of particles
        and is assumed to share the same boundaries when no boundaries or only
        a single set of boundaries are given.
    boundaryfunc : callable or list thereof
        function to correct for finite boundary conditions. Must take three 
        arguments: a numpy array of bin edges, an N×2 numpy array of 
        coordinates and a boundary as specified in `boundary`, and return an 
        N × `len(bin edges)-1` numpy array with a value between 0 and 1 
        specifying the fraction of the area of each circular ring which is 
        within the boundary. See e.g. `_circle_ring_area_frac_in_rectangle`.
    boundary : tuple or list of tuple
        boundary specification as required for boundaryfunc.
    density : float
        average number density of particles in the box to use for normalizing 
        the values of the radial distribution function.
    rmin : float, optional
        lower bound for the pairwise distance, left edge of 0th bin. The 
        default is 0.
    rmax : float, optional
        upper bound for the pairwise distance, right edge of last bin. The 
        default is 10.
    dr : float, optional
        bin width for the pairwise distance bins. The default is (rmax-rmin)/20
    quiet : bool, optional
        if True, no output is printed to the terminal by this function call. 
        The default is False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. The default 
        is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    bincounts : numpy.array
        values for the bins of the radial distribution function
    """
    
    #create bins
    rvals = _get_rvals(rmin, rmax, dr)
    coordinates = _check_coordinate_input(coordinates)
    nf = len(coordinates)
    
    #check boundary and boundaryfunc for type and length
    if boundary is None:
        raise ValueError('boundary must be given when `handle_edge` is not'
                         'one of the default parameters')
    if callable(boundaryfunc):
        boundaryfunc = [boundaryfunc]*nf
        boundary = [boundary]*nf
    elif len(boundary)!=nf or len(boundaryfunc)!=nf:
        raise ValueError('lengths of coordinates, boundary and boundaryfunc '
                         'must match')
    
    #check if density is given and run it if it is a function
    if density is None:
        raise ValueError('density must be given when `handle_edge` is not'
                         'one of the default parameters')
    elif callable(density):
        density = np.mean([density(len(coord),bound) for coord,bound \
                           in zip(coordinates,boundary)])
    elif hasattr(density,'__getitem__') and callable(density[0]):
        density = np.mean([den(len(coord),bound) for den,coord,bound \
                           in zip(density,coordinates,boundary)])
    
    #check if density is a number
    try:
        float(density)
    except TypeError:
        raise ValueError('`density` must be a single numeric value or a '
                         'callable returning one')
    
    #loop over all sets of coordinates
    bincounts = np.zeros(len(rvals)-1)
    for i,(func,bound,coords) in \
        enumerate(zip(boundaryfunc,boundary,coordinates)):
        
        #print progress
        if not quiet:
            print(f'\rcalculating distance histogram g(r) {i+1} of {nf}',
                  end='')

        #set default neighbor_upper_bound
        if type(neighbors_upper_bound)==type(None):
            k = len(coords)
        else:
            k = min([neighbors_upper_bound,len(coords)])
        
        #set up KDTree for fast neighbour finding and query for each particle 
        #separately
        tree = cKDTree(coords)
        dist,indices = tree.query(coords,k=k,distance_upper_bound=rmax,
                                  workers=workers)
    
        #remove pairs with self, padded (infinite) values and anythin below rmin
        dist = dist[:,1:]
        mask = np.isfinite(dist) & (dist>=rmin)

        #when dealing with edges, histogram the distances per reference particle
        #and apply correction factor for missing volume to each particle (each row)
        if _numba_available:
            counts = _apply_hist_nb(dist,mask,rvals)
        else:
            counts = np.zeros((len(dist),len(rvals)-1))
            for j,(row,msk) in enumerate(zip(dist,mask)):
                counts[j] = np.histogram(row[msk],bins=rvals)[0]

        boundarycorr = func(
            rvals,
            coords,
            bound
            )
        counts = np.sum(counts/boundarycorr,axis=0)
        
        #normalize and add to overall list
        #counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
        bincounts += counts / (np.pi * (rvals[1:]**2 - rvals[:-1]**2))\
                         / (density*len(coords))
        
    #newline
    if not quiet:
        print()
    
    #average all datasets
    bincounts /= nf
    
    return rvals,bincounts

def _rdf_dist_hist_3d_cuboid(coordinates,rmin=0,rmax=10,dr=None,boundary=None,
        density=None,periodic_boundary=False,handle_edge=True,quiet=False,
        neighbors_upper_bound=None,workers=1):
    """not intended to be called directly, see top-level `rdf_dist_hist_3d`
    function. This function calculates the rdf in cuboidal (3D 'rectangular')
    boundaries with or without correction for finite size effects, or while 
    accounting for periodic boundary conditions as used in simulations.

    Parameters
    ----------
    coordinates : numpy.array or list-like of numpy.array
        list of sets of coordinates, where each item along the 0th dimension is
        a n*3 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one z-stack, a time step from a 
        video, etc.), with each element of the array of form  `[z,y,x]`. Each 
        set of coordinates is not required to have the same number of particles
        and is assumed to share the same boundaries when no boundaries or only
        a single set of boundaries are given.
    rmin : float, optional
        lower bound for the pairwise distance, left edge of 0th bin. The 
        default is 0.
    rmax : float, optional
        upper bound for the pairwise distance, right edge of last bin. The 
        default is 10.
    dr : float, optional
        bin width for the pairwise distance bins. The default is (rmax-rmin)/20
    boundary : array-like, optional
        positions of the walls that define the bounding box of the coordinates,
        given as  `((zmin,zmax),(ymin,ymax),(xmin,xmax))` if all coordinate 
        sets share the same boundary, or a list of such array-likes of the same
        length as coordinates for specifying boundaries of each set in 
        `coordinates` separately. The default is the min and max values in the 
        dataset along each dimension.
    density : float, optional
        number density of particles in the box to use for normalizing the 
        values. The default is the average density based on `coordinates` and
        `boundary`.
    periodic_boundary : bool, optional
        whether periodic boundary conditions are used. The default is False.
    handle_edge : bool, optional
        whether to correct for edge effects in non-periodic boundary 
        conditions. The default is True.
    quiet : bool, optional
        if True, no output is printed to the terminal by this function call. 
        The default is False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. Note: this is
        ignored when `periodic_boundary=True`. The default is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    bincounts : numpy.array
        values for the bins of the radial distribution function
    
    References
    ----------
    [1] Markus Seserno (2014). How to calculate a three-dimensional g(r) under
    periodic boundary conditions.
    https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf
    
    [2] Kopera, B. A. F., & Retsch, M. (2018). Computing the 3D Radial 
    Distribution Function from Particle Positions: An Advanced Analytic 
    Approach. Analytical Chemistry, 90(23), 13909–13914. 
    https://doi.org/10.1021/acs.analchem.8b03157
    """
    #create bins
    rvals = _get_rvals(rmin, rmax, dr)
    coordinates = _check_coordinate_input(coordinates)
    nf = len(coordinates)
    
    #set default boundary as min and max values in dataset
    if boundary is None:
        userbound = False
        boundary = np.array([[
                [coord[:,0].min(),coord[:,0].max()],
                [coord[:,1].min(),coord[:,1].max()],
                [coord[:,2].min(),coord[:,2].max()]
            ] for coord in coordinates])
    else:
        userbound = True
        boundary = np.array(boundary)
        if boundary.ndim == 2:
            boundary = np.broadcast_to(boundary, (nf,3,2))
    
    #check rmax and boundary for edge-handling in periodic boundary conditions
    if handle_edge:
        if periodic_boundary:
            for bound in boundary:
                if min(bound[:,1]-bound[:,0])==max(bound[:,1]-bound[:,0]):
                    boxlen = bound[0,1]-bound[0,0]
                    if rmax > boxlen*np.sqrt(3)/2:
                        raise ValueError(
                            'rmax cannot be more than sqrt(3)/2 times the size'
                            ' of a cubic bounding box when periodic_boundary='
                            'True, use rmax < '
                            f'{(bound[0,1]-bound[0,0])*np.sqrt(3)/2}'
                        )
                elif rmax > min(bound[:,1]-bound[:,0]):
                    raise NotImplementedError(
                        'rmax larger than half the smallest box dimension when'
                        ' periodic_boundary=True is only implemented for cubic'
                        ' boundaries'
                    )
        
        #check rmax and boundary for edge handling without periodic boundaries
        else:
            for bound in boundary:
                if rmax > max(bound[:,1]-bound[:,0])/2:
                    raise ValueError(
                        'rmax cannot be larger than half the largest dimension'
                        ' in boundary, use rmax < '
                        f'{max(bound[:,1]-bound[:,0])/2}'
                    )
    
    #set density to mean number density in dataset
    if not density:
        vol = np.prod(boundary[:,:,1]-boundary[:,:,0],axis=1)
        density = np.mean([len(coords)/v for v,coords in zip(vol,coordinates)])
    
    #loop over all sets of coordinates
    bincounts = np.zeros(len(rvals)-1)
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #print progress
        if not quiet:
            print(f'\rcalculating distance histogram g(r) {i+1} of {nf}',
                  end='')
        
        #check if coords are within boundary for user given boundary
        if userbound and \
            ((coords<bound[:,0]).any() or (coords>=bound[:,1]).any()):
            
            print()#newline
            warn('not all coordinates are within boundary in coordinate set '
                 f'{i}, ignoring spurious coords',RuntimeWarning)
            coords = coords[
                np.logical_and(
                    coords>=bound[:,0],
                    coords<bound[:,1]
                ).all(axis=1)
            ]
        
        if periodic_boundary:
            #set up KDTree for fast neighbour finding
            #shift box boundary corner to origin for periodic KDTree
            tree = cKDTree(coords-bound[:,0],boxsize=bound[:,1]-bound[:,0])
            
            #count number of particle pairs per bin directly
            counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
            
            #optionally apply edge correction to each bin
            if handle_edge:
                boundarycorr = _sphere_shell_vol_frac_in_cuboid_periodic(
                    rvals,
                    min(bound[:,1]-bound[:,0])
                )
                counts = counts/boundarycorr
            
        #if not periodic_boundary
        else:
            
            #set up KDTree for fast neighbour finding
            tree = cKDTree(coords)
            
            if handle_edge:
                
                #set default neighbor_upper_bound
                if type(neighbors_upper_bound)==type(None):
                    k = len(coords)
                else:
                    k = min([neighbors_upper_bound,len(coords)])
                
                #query tree for any neighbours up to rmax for each particle separately
                dist,indices = tree.query(coords,k=k,distance_upper_bound=rmax,
                                          workers=workers)
            
                #remove pairs with self, padded (infinite) values and anythin below rmin
                dist = dist[:,1:]
                mask = np.isfinite(dist) & (dist>=rmin)
        
                #when dealing with edges, histogram the distances per reference particle
                #and apply correction factor for missing volume to each particle (each row)
                if _numba_available:
                    counts = _apply_hist_nb(dist,mask,rvals)
                else:
                    counts = np.zeros((len(dist),len(rvals)-1))
                    for j,(row,msk) in enumerate(zip(dist,mask)):
                        counts[j] = np.histogram(row[msk],bins=rvals)[0]

                boundarycorr=_sphere_shell_vol_frac_in_cuboid(
                    rvals,
                    bound-coords[:,:,np.newaxis]
                    )
                counts = np.sum(counts/boundarycorr,axis=0)
        
            #otherwise calculate neighbours per bin directly
            else:
                counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
        
        #normalize and add to overall list
        #counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
        bincounts += counts / (4/3*np.pi * (rvals[1:]**3 - rvals[:-1]**3))\
                         / (density*len(coords))
        
    #newline
    if not quiet:
        print()
    
    #average all datasets
    bincounts /= nf
    
    return rvals,bincounts

def _rdf_dist_hist_3d_sphere(coordinates,rmin=0,rmax=10,dr=None,boundary=None,
                     density=None,quiet=False,neighbors_upper_bound=None,
                     workers=1):
    """not intended to be called directly, see top-level `rdf_dist_hist_3d`
    function. This function calculates the rdf in spherical boundary conditions
    while correcting for edge effects.
    
    Parameters
    ----------
    coordinates : numpy.array or list-like of numpy.array
        list of sets of coordinates, where each item along the 0th dimension is
        a n*3 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one z-stack, a time step from a 
        video, etc.), with each element of the array of form  `[z,y,x]`. Each 
        set of coordinates is not required to have the same number of particles
        and is assumed to share the same boundaries when no boundaries or only
        a single set of boundaries are given.
    rmin : float, optional
        lower bound for the pairwise distance, left edge of 0th bin. The 
        default is 0.
    rmax : float, optional
        upper bound for the pairwise distance, right edge of last bin. The 
        default is 10.
    dr : float, optional
        bin width for the pairwise distance bins. The default is (rmax-rmin)/20
    boundary : tuple or list thereof, optional
        origin and radius of the sphere defining the bounding box of the 
        coordinates given as `(z,y,x,radius)` if all coordinate sets share the 
        same boundary, or a list of such tuples of the same length as 
        `coordinates` for specifying boundaries of each set in  `coordinates` 
        separately. The default is the minimum radius sphere centered around
        the mean positions of the particles.
    density : float, optional
        number density of particles in the box to use for normalizing the 
        values. The default is the average density based on `coordinates` and
        `boundary`.
    quiet : bool, optional
        if True, no output is printed to the terminal by this function call. 
        The default is False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. Note: this is
        ignored when `periodic_boundary=True`. The default is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    bincounts : numpy.array
        values for the bins of the radial distribution function
    """
    #create bins
    rvals = _get_rvals(rmin, rmax, dr)
    
    #check coordinate input for type and shape
    coordinates = _check_coordinate_input(coordinates)
    nf = len(coordinates)
    
    #set default boundary origin as mean, rad as max deviation from origin
    if boundary is None:
        boundary = []
        for coord in coordinates:
            bound = np.mean(coord,axis=0)
            rad = np.sqrt(np.sum((coord-bound)**2,axis=1)).max()
            boundary.append([*bound,rad])
        boundary = np.array(boundary)
    else:
        boundary = np.array(boundary)
        if boundary.ndim == 1:
            boundary = np.broadcast_to(boundary, (len(coordinates),4))
    
    if len(boundary) != nf:
                raise ValueError('lengths of coordinates and boundaries must '
                                 'match')
    if any(rmax >= 2*boundary[:,3]):
        raise ValueError('rmax cannot be larger than 2 times the bounding '
                         f'sphere radius, use `rmax<{2*boundary[:,3]}`')
    
    #if density is not given, calculate it
    calc_dens = density is None
    
    #loop over all sets of coordinates
    bincounts = np.zeros(len(rvals)-1)
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #print progress
        if not quiet:
            print(f'\rcalculating distance histogram g(r) {i+1} of {nf}',
                  end='')
        
        #check if coords in boundary, if not raise warning and remove
        d = np.sqrt(np.sum((bound[:3]-coords)**2,axis=1))
        if any(d>bound[3]):
            print()#newline
            warn('not all coordinates are within boundary in coordinate set '
                 f'{i}, ignoring spurious coords',RuntimeWarning)
            coords = coords[d<=bound[3]]
            d = d[d<=bound[3]]
        
        if calc_dens:
            density = len(coords) / (4*np.pi*bound[3]**3/3)
        
        #set up KDTree for fast neighbour finding
        tree = cKDTree(coords)
            
        #set default neighbor_upper_bound
        if type(neighbors_upper_bound)==type(None):
            k = len(coords)
        else:
            k = min([neighbors_upper_bound,len(coords)])
        
        #query tree for any neighbours up to rmax for each particle separately
        dist,indices = tree.query(coords,k=k,distance_upper_bound=rmax,
                                  workers=workers)
    
        #remove pairs with self, padded (infinite) values and anythin below rmin
        dist = dist[:,1:]
        mask = np.isfinite(dist) & (dist>=rmin)

        #histogram the distances per reference particle
        if _numba_available:
            counts = _apply_hist_nb(dist,mask,rvals)
        else:
            counts = np.zeros((len(dist),len(rvals)-1))
            for j,(row,msk) in enumerate(zip(dist,mask)):
                counts[j] = np.histogram(row[msk],bins=rvals)[0]
        
        #apply correction factor for missing volume to each particle (each row)
        boundarycorr=_sphere_shell_vol_frac_in_sphere(
            rvals,
            d,
            bound[3]
            )
        counts = np.sum(counts/boundarycorr,axis=0)

        #normalize and add to overall list
        bincounts += counts / (4/3*np.pi * (rvals[1:]**3 - rvals[:-1]**3))\
                         / (density*len(coords))

    #newline
    if not quiet:
        print()
    
    #average all datasets
    bincounts /= nf
    
    return rvals,bincounts

def _rdf_dist_hist_3d_custom(coordinates,boundaryfunc,boundary,density,rmin=0,
    rmax=10,dr=None,quiet=False,neighbors_upper_bound=None,workers=1):
    """not intended to be called directly, see top-level `rdf_dist_hist_3d`
    function. This function calculates the rdf using boundaries and corrections
    supplied via arguments so that custom boundary geometries can be corrected
    for, as well as for a list of datasets with different boundary coditions
    between them.

    Parameters
    ----------
    coordinates : numpy.array or list-like of numpy.array
        list of sets of coordinates, where each item along the 0th dimension is
        a n*3 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one z-stack, a time step from a 
        video, etc.), with each element of the array of form  `[z,y,x]`. Each 
        set of coordinates is not required to have the same number of particles
        and is assumed to share the same boundaries when no boundaries or only
        a single set of boundaries are given.
    boundaryfunc : callable or list thereof
        function to correct for finite boundary conditions. Must take three 
        arguments: a numpy array of bin edges, an N×2 numpy array of 
        coordinates and a boundary as specified in `boundary`, and return an 
        N × `len(bin edges)-1` numpy array with a value between 0 and 1 
        specifying the fraction of the area of each circular ring which is 
        within the boundary. See e.g. `_sphere_shell_vol_frac_in_cuboid`.
    boundary : tuple or list of tuple
        boundary specification as required for boundaryfunc.
    density : float
        average number density of particles in the box to use for normalizing 
        the values of the radial distribution function.
    rmin : float, optional
        lower bound for the pairwise distance, left edge of 0th bin. The 
        default is 0.
    rmax : float, optional
        upper bound for the pairwise distance, right edge of last bin. The 
        default is 10.
    dr : float, optional
        bin width for the pairwise distance bins. The default is (rmax-rmin)/20
    quiet : bool, optional
        if True, no output is printed to the terminal by this function call. 
        The default is False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. Note: this is
        ignored when `periodic_boundary=True`. The default is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    bincounts : numpy.array
        values for the bins of the radial distribution function
    """

    #create bins
    rvals = _get_rvals(rmin, rmax, dr)
    coordinates = _check_coordinate_input(coordinates)
    nf = len(coordinates)
    
    #check boundary and boundaryfunc for type and length
    if boundary is None:
        raise ValueError('boundary must be given when `handle_edge` is not'
                         'one of the default parameters')
    if callable(boundaryfunc):
        boundaryfunc = [boundaryfunc]*nf
        boundary = [boundary]*nf
    elif len(boundary)!=nf or len(boundaryfunc)!=nf:
        raise ValueError('lengths of coordinates, boundary and boundaryfunc '
                         'must match')
    
    #check if density is given and run it if it is a function
    if density is None:
        raise ValueError('density must be given when `handle_edge` is not'
                         'one of the default parameters')
    elif callable(density):
        density = np.mean([density(len(coord),bound) for coord,bound \
                           in zip(coordinates,boundary)])
    
    #check if density is a number
    try:
        float(density)
    except ValueError:
        raise ValueError('`density` must be a single numeric value or a '
                         'callable returning one')
    
    #loop over all sets of coordinates
    bincounts = np.zeros(len(rvals)-1)
    for i,(func,bound,coords) in \
        enumerate(zip(boundaryfunc,boundary,coordinates)):
        
        #print progress
        if not quiet:
            print(f'\rcalculating distance histogram g(r) {i+1} of {nf}',
                  end='')

        #set default neighbor_upper_bound
        if type(neighbors_upper_bound)==type(None):
            k = len(coords)
        else:
            k = min([neighbors_upper_bound,len(coords)])
        
        #set up KDTree for fast neighbour finding and query for each particle 
        #separately
        tree = cKDTree(coords)
        dist,indices = tree.query(coords,k=k,distance_upper_bound=rmax,
                                  workers=workers)
    
        #remove pairs with self, padded (infinite) values and anythin below rmin
        dist = dist[:,1:]
        mask = np.isfinite(dist) & (dist>=rmin)

        #when dealing with edges, histogram the distances per reference particle
        #and apply correction factor for missing volume to each particle (each row)
        if _numba_available:
            counts = _apply_hist_nb(dist,mask,rvals)
        else:
            counts = np.zeros((len(dist),len(rvals)-1))
            for j,(row,msk) in enumerate(zip(dist,mask)):
                counts[j] = np.histogram(row[msk],bins=rvals)[0]

        boundarycorr = func(
            rvals,
            coords,
            bound
            )
        counts = np.sum(counts/boundarycorr,axis=0)
        
        #normalize and add to overall list
        #counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
        bincounts += counts / (4/3*np.pi * (rvals[1:]**3 - rvals[:-1]**3))\
                         / (density*len(coords))
        
    #newline
    if not quiet:
        print()
    
    #average all datasets
    bincounts /= nf
    
    return rvals,bincounts

#%% test-particle insertion functions

def _rdf_insertion_binned_2d_rectangle(coordinates,pairpotential,rmin=0,
    rmax=10,dr=None,boundary=None,pairpotential_binedges=None,n_ins=1000,
    interpolate=True,periodic_boundary=False,avoid_boundary=False,
    avoid_coordinates=False,neighbors_upper_bound=None,workers=1):
    """not intended to be called directly, see top-level `rdf_dist_hist_2d`
    function. This function calculates the rdf from insertion of test-particles
    into sets of existing 2D coordinates in rectangular boundaries, averaged 
    over bins of width dr, and based on the pairwise interaction potential u(r)
    (in units of kT).

    Parameters
    ----------
    coordinates : list-like of numpy.array
        List of sets of coordinates, where each item along the 0th dimension is
        a n*2 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one time step from a video, etc.),
        with each element of the array of form  `[y,x]`. Each  set of 
        coordinates is not required to have the same number of particles but 
        all coordinates must be within the bounding box(es) given in 
        `boundary`.
    pairpotential : iterable
        list of values for the pairwise interaction potential. Must have length
        of `len(pairpotential_binedges)-1` and be in units of thermal energy kT
    rmin : float, optional
        lower cut-off for the pairwise distance. The default is 0.
    rmax : float
        cut-off radius for the pairwise distance (right edge of last bin).
    dr : float
        bin width of the pairwise distance bins.
    boundary : array-like of form `((ymin,ymax),(xmin,xmax))`
        positions of the walls that define the bounding box of the coordinates,
        given as a single array-like for a shared set of boundaries for all 
        coordinates, or an list-like of such array-likes with the same length 
        as `coordinates` for a separate set of boundaries for each.
    pairpotential_binedges : iterable, optional
        bin edges corresponding to the values in `pairpotential. The default 
        is None, which uses the bins defined by `rmin`, `rmax` and `dr`.
    n_ins : int, optional
        the number of test-particles to insert into each item in `coordinates`.
        The default is 1000.
    interpolate : bool, optional
        whether to use linear interpolation for calculating the interaction of 
        two particles using the values in `pairpotential`. The default is True.
        If False, the nearest bin value is used.
    periodic_boundary : bool, optional
        whether periodic boundary conditions are used. The default is False.
    avoid_boundary : bool, optional
        if True, all test-particles are inserted at least `rmax` away from any 
        of the surfaces defined in `boundary` to avoid effects of the finite
        volume of the bounding box. The default is False, which uses an 
        analytical correction factor for missing volume of test-particles near 
        the boundaries.
    avoid_coordinates : bool, optional
        whether to insert test-particles at least `rmin` away from the center 
        of any of the 'real' coordinates in `coordinates`. The default is 
        False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. Note: this is
        ignored when `periodic_boundary=True`. The default is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    pair_correlation : numpy.array
        values for the rdf / pair correlation function in each bin
    counter : numpy.array
        number of pair counts that contributed to the (mean) values in each bin
    """
    #create bin edges and bin centres for r
    rvals = _get_rvals(rmin, rmax, dr)
    rcent = (rvals[1:]+rvals[:-1])/2
    nf = len(coordinates)
    nr = len(rcent)
    
    coordinates = _check_coordinate_input(coordinates)
    
    #set default boundary as min and max values in dataset
    if boundary is None:
        boundary = np.array([[
                [coord[:,0].min(),coord[:,0].max()],
                [coord[:,1].min(),coord[:,1].max()]
            ] for coord in coordinates])
    else:
        boundary = np.array(boundary)
        if boundary.ndim == 2:
            boundary = np.broadcast_to(boundary, (nf,2,2))
    
    #check if rmax input and boundary are feasible for avoiding boundary
    for bound in boundary:
        if avoid_boundary and rmax >= min(bound[:,1]-bound[:,0])/2:
            raise ValueError(
                'rmax cannot be more than half the smallest box dimension when'
                ' avoid_boundary=True, use '
                f'rmax<{min(bound[:,1]-bound[:,0])/2}'
            )
        
        #check rmax and boundary for edge-handling in periodic boundary conditions
        elif periodic_boundary:
            if bound[0,1]-bound[0,0] == bound[1,1]-bound[1,0]:
                boxlen = bound[0,1]-bound[0,0]
                if rmax > boxlen*np.sqrt(2)/2:
                    raise ValueError(
                        'rmax cannot be more than sqrt(2)/2 times the size of '
                        'a square bounding box when using periodic boundary '
                        'conditions, use rmax<'
                        f'{(bound[0,1]-bound[0,0])*np.sqrt(2)/2}'
                    )
            elif rmax > min(bound[:,1]-bound[:,0]):
                raise NotImplementedError(
                    'rmax larger than half the smallest box dimension when '
                    'using periodic boundaries is only implemented for square '
                    'boundaries'
                )
        
        #check rmax and boundary for edge handling without periodic boundaries
        else:
            if rmax > max(bound[:,1]-bound[:,0])/2:
                raise ValueError(
                    'rmax cannot be larger than half the largest dimension in '
                    'boundary, use rmax<{max(bound[:,1]-bound[:,0])/2}'
                )
    
    #init function that returns energy from list of pairwise distances
    #if function is given, use that
    if callable(pairpotential):
        pot_fun = pairpotential
    #otherwise use nearest or linearly interpolate
    else:
        #bin edges and centres for pairpotential
        if pairpotential_binedges is None:
            pairpotential_binedges = rvals
        pairpotential_bincenter = (pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2
        
        #convert bool values of interpolate to nearest (no interpolation) and linear
        if type(interpolate)==bool:
            if interpolate:
                interpolate='linear'
            else:
                interpolate='nearest'

        from scipy.interpolate import interp1d   
        pot_fun = interp1d(
            pairpotential_bincenter,
            pairpotential,
            kind=interpolate,
            bounds_error=False,
            fill_value='extrapolate'
        )
    
    #define a reduced area for test-particles away from all boundaries
    if avoid_boundary:
        reduced_boundary = boundary.copy()
        reduced_boundary[:,:,0] += rmax
        reduced_boundary[:,:,1] -= rmax
    
    #initialize arrays to store values
    counter = np.zeros(nr)
    pair_correlation = np.zeros(nr)
    
    #loop over all timesteps / independent sets of coordiates
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #check if coords are within boundary
        if (coords<bound[:,0]).any() or (coords>bound[:,1]).any():
            print()#newline
            warn('not all coordinates are within boundary in coordinate set '
                 f'{i}, ignoring spurious coords',RuntimeWarning)
            coords = coords[
                np.logical_and(
                    coords>=bound[:,0],
                    coords<bound[:,1]
                ).all(axis=1)
            ]
        
        #generate new test-particle coordinates for each set
        if avoid_coordinates and avoid_boundary:
            trialparticles = _rand_coord_at_dist(reduced_boundary[i],coords,
                                                 rmin,n=n_ins)
        elif avoid_coordinates:
            trialparticles = _rand_coord_at_dist(bound,coords,rmin,n=n_ins)
        elif avoid_boundary:
            trialparticles = _rand_coord_in_box(reduced_boundary[i],n=n_ins)
        else:
            trialparticles = _rand_coord_in_box(bound,n=n_ins)
         
        #init KDTree for fast pairfinding
        if periodic_boundary:
            coords -= bound[:,0]#shift box to origin
            tree = cKDTree(coords,boxsize=bound[:,1]-bound[:,0])
        else:
            tree = cKDTree(coords)
        
        #set default neighbor_upper_bound
        if type(neighbors_upper_bound)==type(None):
            k = len(coords)
        else:
            k = min([neighbors_upper_bound,len(coords)])
        
        #find all pairs with one particle from testparticles and one from coordinates
        distances,_ = tree.query(trialparticles,k=k,distance_upper_bound=rmax,
                                 workers=workers)
        
        #cKDTree pads rows with np.inf to get correct length, work with masked
        #arrays to only work on finite values
        mask = np.isfinite(distances) & (distances>0)
        
        #calculate total potential energy of insertion (psi) for each testparticle
        #(row) by summing each pairwise potential energy u(r)
        if avoid_boundary:
            exp_psi = np.empty(n_ins)
            for j,(row,msk) in enumerate(zip(distances,mask)):
                exp_psi[j] = np.exp(-np.sum(pot_fun(row[msk])))
        
        else:
            if periodic_boundary:
                #calculate correction factor for each distance bin to account 
                # for missing information
                boundarycorr = _circle_ring_area_frac_in_rectangle_periodic(
                    rvals,
                    min(bound[:,1]-bound[:,0])
                )[np.newaxis,:]
                
            else:
                #calculate correction factor for each testparticle for each dist bin
                # to account for missing information around particles near boundary.
                #boundary is shifted for coordinate system with origin in particle
                boundarycorr = _circle_ring_area_frac_in_rectangle(
                    rvals,
                    bound-trialparticles[:,:,np.newaxis]
                )
                
            #sum pairwise energy per particle per distance bin, then correct
            # each bin for missing volume, then sum and convert to probability 
            # e^(-psi)
            exp_psi = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                dist = row[msk]
                exp_psi[j] = np.histogram(dist,bins=rvals,weights=pot_fun(dist))[0]
            
            #calculate probability of testparticles
            exp_psi = np.exp(-np.sum(exp_psi/boundarycorr,axis=1))
            
        #calculate average probability of all testparticles in set
        #prob_tot = np.mean(exp_psi)
        
        #count how many pairs each particle contributes to each bin
        if _numba_available:
            counts = _apply_hist_nb(distances,mask,rvals)
        else:
            counts = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                counts[j] = np.histogram(row[msk],bins=rvals)[0]
        
        #take average insertion probability of all test-particles
        prob_tot = np.mean(exp_psi)
        
        #boundary correct counts if needed
        if avoid_boundary:
            corrcounts = counts.copy()
        else:
            corrcounts = counts.copy() / boundarycorr
        
        #count up the insertion probabilities counting towards each bin using 
        #the edge-corrected count number so that all distances have equal 
        #weight with respect to the averaging in prob_tot
        prob_r = np.sum(corrcounts*exp_psi[:,np.newaxis],axis=0)
        corrcounts = corrcounts.sum(axis=0)
        prob_r[corrcounts!=0] /= corrcounts[corrcounts!=0]
        
        #add to overall lists weighted by total count
        pair_correlation += counts.sum() * prob_r / prob_tot
        counter += counts.sum(axis=0)
      
    #take total count weighted average of all datasets
    pair_correlation /= counter.sum()
    pair_correlation[counter==0] = 0
    
    return rvals,pair_correlation,counter


def _rdf_insertion_binned_2d_circle(coordinates,pairpotential,rmin=0,rmax=20,
        dr=None,boundary=None,pairpotential_binedges=None,n_ins=1000,
        interpolate=True,avoid_boundary=False,neighbors_upper_bound=None,
        workers=1):
    """not intended to be called directly, see top-level `rdf_dist_hist_2d`
    function. This function calculates the rdf from insertion of test-particles
    into sets of existing 2D coordinates in circular boundaries, averaged over 
    bins of width dr, and based on the pairwise interaction potential u(r) (in 
    units of kT).

    Parameters
    ----------
    coordinates : list-like of numpy.array
        List of sets of coordinates, where each item along the 0th dimension is
        a n*2 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one time step from a video, etc.),
        with each element of the array of form  `[y,x]`. Each  set of 
        coordinates is not required to have the same number of particles but 
        all coordinates must be within the bounding box(es) given in 
        `boundary`.
    pairpotential : iterable
        list of values for the pairwise interaction potential. Must have length
        of `len(pairpotential_binedges)-1` and be in units of thermal energy kT
    rmin : float, optional
        lower cut-off for the pairwise distance. The default is 0.
    rmax : float
        cut-off radius for the pairwise distance (right edge of last bin).
    dr : float
        bin width of the pairwise distance bins.
    boundary : array-like of form `(y,x,r)`
        positions of the walls that define the bounding box of the coordinates,
        given as a single array-like for a shared set of boundaries for all 
        coordinates, or an list-like of such array-likes with the same length 
        as `coordinates` for a separate set of boundaries for each.
    pairpotential_binedges : iterable, optional
        bin edges corresponding to the values in `pairpotential. The default 
        is None, which uses the bins defined by `rmin`, `rmax` and `dr`.
    n_ins : int, optional
        the number of test-particles to insert into each item in `coordinates`.
        The default is 1000.
    interpolate : bool, optional
        whether to use linear interpolation for calculating the interaction of 
        two particles using the values in `pairpotential`. The default is True.
        If False, the nearest bin value is used.
    periodic_boundary : bool, optional
        whether periodic boundary conditions are used. The default is False.
    avoid_boundary : bool, optional
        if True, all test-particles are inserted at least `rmax` away from any 
        of the surfaces defined in `boundary` to avoid effects of the finite
        volume of the bounding box. The default is False, which uses an 
        analytical correction factor for missing volume of test-particles near 
        the boundaries.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. The default 
        is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    pair_correlation : numpy.array
        values for the rdf / pair correlation function in each bin
    counter : numpy.array
        number of pair counts that contributed to the (mean) values in each bin
    """
    #create bin edges and bin centres for r
    coordinates = _check_coordinate_input(coordinates)
    rvals = _get_rvals(rmin, rmax, dr)
    rcent = (rvals[1:]+rvals[:-1])/2
    nf = len(coordinates)
    nr = len(rcent)

    #set default boundary or make sure the length is correct
    if boundary is None:
        boundary = []
        for coord in coordinates:
            bound = np.mean(coord,axis=0)
            rad = np.sqrt(np.sum((coord-bound)**2,axis=1)).max()
            boundary.append(np.concatenate([bound,rad]))
        boundary = np.array(boundary)
    else:
        boundary = np.array(boundary)
        if boundary.ndim == 2:
            boundary = np.broadcast_to(boundary, (nf,3))

    if len(boundary) != nf:
                raise ValueError('lengths of coordinates and boundaries must '
                                 'match')
    if any(rmax >= 2*boundary[:,2]):
        raise ValueError('rmax cannot be larger than 2 times the bounding '
                         f'circle radius, use `rmax<{2*boundary[:,2]}`')

    #init function that returns energy from list of pairwise distances
    #if function is given, use that
    if callable(pairpotential):
        pot_fun = pairpotential
    #otherwise use nearest or linearly interpolate
    else:
        #bin edges and centres for pairpotential
        if pairpotential_binedges is None:
            pairpotential_binedges = rvals
        pairpotential_bincenter = \
            (pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2
        
        #convert bool values of interpolate to nearest or linear
        if type(interpolate)==bool:
            if interpolate:
                interpolate='linear'
            else:
                interpolate='nearest'

        from scipy.interpolate import interp1d   
        pot_fun = interp1d(
            pairpotential_bincenter,
            pairpotential,
            kind=interpolate,
            bounds_error=False,
            fill_value='extrapolate'
        )
    
    #initialize arrays to store values
    counter = np.zeros(nr)
    pair_correlation = np.zeros(nr)
    
    #loop over all timesteps / independent sets of coordiates
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #generate new test-particle coordinates for each set
        if avoid_boundary:
            trialparticles = _rand_coord_in_circle(bound[:2],bound[2]-rmax,n=n_ins)
        else:
            trialparticles = _rand_coord_in_circle(bound[:2],bound[2],n=n_ins)
         
        #init KDTree for fast pairfinding
        tree = cKDTree(coords)
        
        #find all pairs with one particle from testparticles and one from coordinates
        distances,_ = tree.query(trialparticles,k=len(coords),
                                 distance_upper_bound=rmax,workers=workers)
        
        #cKDTree pads rows with np.inf to get correct length, work with masked
        #arrays to only work on finite values
        mask = np.isfinite(distances) & (distances>0)
        distances = np.ma.masked_array(distances,mask)
        
        #calculate total potential energy of insertion (psi) for each testparticle
        #(row) by summing each pairwise potential energy u(r)
        if avoid_boundary:
            exp_psi = np.empty(n_ins)
            for j,(row,msk) in enumerate(zip(distances,mask)):
                exp_psi[j] = np.exp(-np.sum(pot_fun(row[msk])))
        
        else:
            #calculate correction factor for each testparticle for each dist bin
            # to account for missing information around particles near boundary.
            boundarycorr = _circle_ring_area_frac_in_circle(
                rvals,
                np.sqrt(np.sum((trialparticles - np.array(bound[:2]))**2,axis=1)),
                bound[2]
            )
                
            #sum pairwise energy per particle per distance bin, then correct
            # each bin for missing volume, then sum and convert to probability 
            # e^(-psi)
            exp_psi = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                dist = row[msk]
                exp_psi[j] = np.histogram(dist,bins=rvals,weights=pot_fun(dist))[0]
            
            #calculate probability of testparticles
            exp_psi = np.exp(-np.sum(exp_psi/boundarycorr,axis=1))
            
        #count how many pairs each particle contributes to each bin
        if _numba_available:
            counts = _apply_hist_nb(distances,mask,rvals)
        else:
            counts = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                counts[j] = np.histogram(row[msk],bins=rvals)[0]
        
        #take average insertion probability of all test-particles
        prob_tot = np.mean(exp_psi)
        
        #boundary correct counts if needed
        if avoid_boundary:
            corrcounts = counts.copy()
        else:
            corrcounts = counts.copy() / boundarycorr
        
        #count up the insertion probabilities counting towards each bin using 
        #the edge-corrected count number so that all distances have equal 
        #weight with respect to the averaging in prob_tot
        prob_r = np.sum(corrcounts*exp_psi[:,np.newaxis],axis=0)
        corrcounts = corrcounts.sum(axis=0)
        prob_r[corrcounts!=0] /= corrcounts[corrcounts!=0]
        
        #add to overall lists weighted by total count
        pair_correlation += counts.sum() * prob_r / prob_tot
        counter += counts.sum(axis=0)
      
    #take total count weighted average of all datasets
    pair_correlation /= counter.sum()
    pair_correlation[counter==0] = 0
    
    return rvals,pair_correlation,counter

def _rdf_insertion_binned_2d_custom(coordinates,pairpotential,boundary_func,
    boundary,testparticle_func,rmin=0,rmax=10,dr=None,
    pairpotential_binedges=None,n_ins=1000,interpolate=True,
    avoid_coordinates=False,neighbors_upper_bound=None,workers=1):
    """not intended to be called directly, see top-level `rdf_dist_hist_2d`
    function. This function calculates the rdf from insertion of test-particles
    using boundaries and corrections supplied via arguments so that custom 
    boundary geometries can be corrected for, as well as for a list of datasets
    with different boundary coditions between them, averaged over bins of width
    dr, and based on the pairwise interaction potential u(r) (in units of kT).
    
    Parameters
    ----------
    coordinates : list-like of numpy.array
        List of sets of coordinates, where each item along the 0th dimension is
        a n*2 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one time step from a video, etc.),
        with each element of the array of form  `[y,x]`. Each  set of 
        coordinates is not required to have the same number of particles but 
        all coordinates must be within the bounding box(es) given in 
        `boundary`.
    pairpotential : iterable
        list of values for the pairwise interaction potential. Must have length
        of `len(pairpotential_binedges)-1` and be in units of thermal energy kT
    boundaryfunc : callable or list thereof
        function to correct for finite boundary conditions. Must take three 
        arguments: a numpy array of bin edges, an N×2 numpy array of 
        coordinates and a boundary as specified in `boundary`, and return an 
        N × `len(bin edges)-1` numpy array with a value between 0 and 1 
        specifying the fraction of the area of each circular ring which is 
        within the boundary. See e.g. `_circle_ring_area_frac_in_rectangle`.
    boundary : tuple or list thereof
        boundary specification as required for boundaryfunc.
    testparticle_func : callable or list thereof
        function that randomly generates testparticles within the boundary. 
        Must take 2 arguments, resp. the boundary and n_ins, or 4 argumants, 
        resp. the boundaries, coordinates, minimum distance and n_ins when
        `avoid_coordinates=True`. See e.g. `_rand_coord_in_box` or 
        `_rand_coord_at_dist`.
    rmin : float, optional
        lower cut-off for the pairwise distance. The default is 0.
    rmax : float
        cut-off radius for the pairwise distance (right edge of last bin).
    dr : float
        bin width of the pairwise distance bins.
    boundary : array-like of form `((ymin,ymax),(xmin,xmax))`
        positions of the walls that define the bounding box of the coordinates,
        given as a single array-like for a shared set of boundaries for all 
        coordinates, or an list-like of such array-likes with the same length 
        as `coordinates` for a separate set of boundaries for each.
    pairpotential_binedges : iterable, optional
        bin edges corresponding to the values in `pairpotential. The default 
        is None, which uses the bins defined by `rmin`, `rmax` and `dr`.
    n_ins : int, optional
        the number of test-particles to insert into each item in `coordinates`.
        The default is 1000.
    interpolate : bool, optional
        whether to use linear interpolation for calculating the interaction of 
        two particles using the values in `pairpotential`. The default is True.
        If False, the nearest bin value is used.
    avoid_coordinates : bool, optional
        whether to insert test-particles at least `rmin` away from the center 
        of any of the 'real' coordinates in `coordinates`. The default is 
        False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. The default 
        is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    pair_correlation : numpy.array
        values for the rdf / pair correlation function in each bin
    counter : numpy.array
        number of pair counts that contributed to the (mean) values in each bin
    
    References
    ----------
    [1] Stones, A. E., Dullens, R. P. A., & Aarts, D. G. A. L. (2019). Model-
    Free Measurement of the Pair Potential in Colloidal Fluids Using Optical 
    Microscopy. Physical Review Letters, 123(9), 098002. 
    https://doi.org/10.1103/PhysRevLett.123.098002
    
    """
    _check_coordinate_input(coordinates)
    
    #create bin edges and bin centres for r
    rvals = _get_rvals(rmin, rmax, dr)
    rcent = (rvals[1:]+rvals[:-1])/2
    nf = len(coordinates)
    nr = len(rcent)
    
    #check if testparticle_func is given
    if testparticle_func is None:
        raise ValueError('testparticle_func must be given when using custom '
                         'boundary handling')
    
    #make sure boundary is given and is list
    if boundary is None:
        raise ValueError('when using custom boundary handling, `boundary` '
                         'must be given')
    elif type(boundary) != list:
        boundary = [boundary]*nf
    
    #assure lists for functions
    if callable(boundary_func) and callable(testparticle_func):
        boundary_func = [boundary_func]*nf
        testparticle_func = [testparticle_func]*nf
    
    #can't mix list and single func
    elif callable(boundary_func) or callable(testparticle_func):
        raise TypeError('either a single boundary, boundary_func and '
                        'testparticle_func must be given, or all must be a '
                        'list matching the length of coordinates')
    
    #check lengths
    if len(coordinates) != len(boundary):
        raise ValueError('lengths of `boundary` and `coordinates` must match')
    if len(coordinates) != len(boundary_func):
        raise ValueError('lengths of boundary correction functions and '
                         '`coordinates` must match')
    if len(coordinates) != len(testparticle_func):
        raise ValueError('lengths of `testparticle_func` and `coordinates` '
                         'must match')
    
    #init function that returns energy from list of pairwise distances
    #if function is given, use that
    if callable(pairpotential):
        pot_fun = pairpotential
    #otherwise use nearest or linearly interpolate
    else:
        #bin edges and centres for pairpotential
        if pairpotential_binedges is None:
            pairpotential_binedges = rvals
        pairpotential_bincenter = \
            (pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2
        
        #convert bool values of interpolate to nearest or linear
        if type(interpolate)==bool:
            if interpolate:
                interpolate='linear'
            else:
                interpolate='nearest'

        from scipy.interpolate import interp1d   
        pot_fun = interp1d(
            pairpotential_bincenter,
            pairpotential,
            kind=interpolate,
            bounds_error=False,
            fill_value='extrapolate'
        )
    
    #initialize arrays to store values
    counter = np.zeros(nr)
    pair_correlation = np.zeros(nr)
    
    #loop over all timesteps / independent sets of coordiates
    for i,(bound,boundfunc,testfunc,coords) in \
        enumerate(zip(boundary,boundary_func,testparticle_func,coordinates)):
        
        #generate new test-particle coordinates for each set
        if avoid_coordinates:
            trialparticles = testfunc(bound,coords,rmin,n_ins)
        else:
            trialparticles = testfunc(bound,n_ins)
        
        #init KDTree for fast pairfinding
        tree = cKDTree(coords)
        
        #set default neighbor_upper_bound
        if type(neighbors_upper_bound)==type(None):
            k = len(coords)
        else:
            k = min([neighbors_upper_bound,len(coords)])
        
        #find all pairs with one particle from testparticles and one from coordinates
        distances,_ = tree.query(trialparticles,k=k,distance_upper_bound=rmax,
                                 workers=workers)
        
        #cKDTree pads rows with np.inf to get correct length, work with masked
        #arrays to only work on finite values
        mask = np.isfinite(distances) & (distances>0)
        distances = np.ma.masked_array(distances,mask)
        
        #calculate total potential energy of insertion (psi) for each testparticle
        #(row) by summing each pairwise potential energy u(r) 
        #calculate correction factor for each testparticle for each dist bin
        # to account for missing information around particles near boundary.
        #boundary is shifted for coordinate system with origin in particle
        boundarycorr = boundfunc(
            rvals,
            trialparticles,
            bound
        )
                
        #sum pairwise energy per particle per distance bin, then correct
        # each bin for missing volume, then sum and convert to probability 
        # e^(-psi)
        exp_psi = np.empty((n_ins,nr))
        for j,(row,msk) in enumerate(zip(distances,mask)):
            dist = row[msk]
            exp_psi[j] = np.histogram(dist,bins=rvals,weights=pot_fun(dist))[0]
            
        #calculate probability of testparticles
        exp_psi = np.exp(-np.sum(exp_psi/boundarycorr,axis=1))
            
        #count how many pairs each particle contributes to each bin
        if _numba_available:
            counts = _apply_hist_nb(distances,mask,rvals)
        else:
            counts = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                counts[j] = np.histogram(row[msk],bins=rvals)[0]
        
        #take average insertion probability of all test-particles
        prob_tot = np.mean(exp_psi)
        
        #boundary correct counts
        corrcounts = counts.copy() / boundarycorr
        
        #count up the insertion probabilities counting towards each bin using 
        #the edge-corrected count number so that all distances have equal 
        #weight with respect to the averaging in prob_tot
        prob_r = np.sum(corrcounts*exp_psi[:,np.newaxis],axis=0)
        corrcounts = corrcounts.sum(axis=0)
        prob_r[corrcounts!=0] /= corrcounts[corrcounts!=0]
        
        #add to overall lists weighted by total count
        pair_correlation += counts.sum() * prob_r / prob_tot
        counter += counts.sum(axis=0)
      
    #take total count weighted average of all datasets
    pair_correlation /= counter.sum()
    pair_correlation[counter==0] = 0
    
    return rvals,pair_correlation,counter

def _rdf_insertion_binned_3d_cuboid(coordinates,pairpotential,rmin=0,rmax=10,
    dr=None,boundary=None,pairpotential_binedges=None,n_ins=1000,
    interpolate=True,periodic_boundary=False,ins_coords=None,insert_grid=False,
    avoid_boundary=False,avoid_coordinates=False,neighbors_upper_bound=None,
    workers=1):
    """not intended to be called directly, see top-level `rdf_dist_hist_3d`
    function. This function calculates the rdf from insertion of test-particles
    into sets of existing 3D coordinates in cuboidal boundaries, averaged over 
    bins of width dr, and based on the pairwise interaction potential u(r) (in 
    units of kT).

    Parameters
    ----------
    coordinates : list-like of numpy.array
        List of sets of coordinates, where each item along the 0th dimension is
        a n*3 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one z-stack, a time step from a 
        video, etc.), with each element of the array of form  `[z,y,x]`. Each 
        set of coordinates is not required to have the same number of particles
        but all coordinates must be within the bounding box(es) given in 
        `boundary`.
    pairpotential : iterable or callable
        list of values for the pairwise interaction potential. Must have length
        of `len(pairpotential_binedges)-1` and be in units of thermal energy 
        kT. Alternatively, any callable (function) which takes only a numpy 
        array of pairwise distances and returns the pairpotential can be given.
    rmin : float, optional
        lower cut-off for the pairwise distance. The default is 0.
    rmax : float
        cut-off radius for the pairwise distance (right edge of last bin).
    dr : float
        bin width of the pairwise distance bins.
    boundary : array-like of form `((zmin,zmax),(ymin,ymax),(xmin,xmax))`
        Positions of the walls that define the bounding box of the coordinates,
        given as a single array-like for a shared set of boundaries for all 
        coordinates, or an list-like of such array-likes with the same length 
        as `coordinates` for a separate set of boundaries for each.
    pairpotential_binedges : iterable, optional
        bin edges corresponding to the values in `pairpotential. The default 
        is None, which uses the bins defined by `rmin`, `rmax` and `dr`. This 
        parameter is ignored if pairpotential is a callable (function)
    n_ins : int, optional
        the number of test-particles to insert into each item in `coordinates`.
        The default is 1000.
    interpolate : bool, optional
        whether to use linear interpolation for calculating the interaction of 
        two particles using the values in `pairpotential`. The default is True.
        If False, the nearest bin value is used. This parameter is ignored if 
        pairpotential is a callable (function).
    periodic_boundary : bool, optional
        whether periodic boundary conditions are used. The default is False.
    insert_grid : bool, optional
        Wheter to insert the coordinates on an evenly spaced regular grid. The 
        default is False which inserts on uniformly distributed pseudorandom
        coordinates.
    avoid_boundary : bool, optional
        if True, all test-particles are inserted at least `rmax` away from any 
        of the surfaces defined in `boundary` to avoid effects of the finite
        volume of the bounding box. The default is False, which uses an 
        analytical correction factor for missing volume of test-particles near 
        the boundaries.
    avoid_coordinates : bool, optional
        whether to insert test-particles at least `rmin` away from the center 
        of any of the 'real' coordinates in `coordinates`. The default is 
        False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. Note: this is
        ignored when `periodic_boundary=True`. The default is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    pair_correlation : numpy.array
        values for the rdf / pair correlation function in each bin
    counter : numpy.array
        number of pair counts that contributed to the (mean) values in each bin
    
    References
    ----------
    [1] Markus Seserno (2014). How to calculate a three-dimensional g(r) under
    periodic boundary conditions.
    https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf
    
    [2] Kopera, B. A. F., & Retsch, M. (2018). Computing the 3D Radial 
    Distribution Function from Particle Positions: An Advanced Analytic 
    Approach. Analytical Chemistry, 90(23), 13909–13914. 
    https://doi.org/10.1021/acs.analchem.8b03157
    """
    #create bin edges and bin centres for r
    coordinates = _check_coordinate_input(coordinates)
    rvals = _get_rvals(rmin, rmax, dr)
    rcent = (rvals[1:]+rvals[:-1])/2
    nr = len(rcent)
    nf = len(coordinates)
    
    #set default boundary as min and max values in dataset
    if boundary is None:
        boundary = np.array([[
                [coord[:,0].min(),coord[:,0].max()],
                [coord[:,1].min(),coord[:,1].max()],
                [coord[:,2].min(),coord[:,2].max()]
            ] for coord in coordinates])
    else:
        boundary = np.array(boundary)
        if boundary.ndim == 2:
            boundary = np.broadcast_to(boundary, (nf,3,2))
        
    #check if rmax input and boundary are feasible for avoiding boundary
    for bound in boundary:
        if avoid_boundary and rmax >= min(bound[:,1]-bound[:,0])/2:
            raise ValueError(
                'rmax cannot be more than half the smallest box dimension when'
                ' avoid_boundary=True, use rmax < '
                f'{min(bound[:,1]-bound[:,0])/2}')
    
        #check rmax and boundary for edge-handling in periodic boundary conditions
        elif periodic_boundary:
            if min(bound[:,1]-bound[:,0])==max(bound[:,1]-bound[:,0]):
                boxlen = bound[0,1]-bound[0,0]
                if rmax > boxlen*np.sqrt(3)/2:
                    raise ValueError(
                        'rmax cannot be more than sqrt(3)/2 times the size of '
                        'a cubic bounding box when periodic_boundary=True, use'
                        f' rmax < {(bound[0,1]-bound[0,0])*np.sqrt(3)/2}'
                    )
            elif rmax > min(bound[:,1]-bound[:,0]):
                raise NotImplementedError(
                    'rmax larger than half the smallest box dimension when '
                    '`periodic_boundary=True` is only implemented for cubic '
                    'boundaries'
                )
    
        #check rmax and boundary for edge handling without periodic boundaries
        else:
            if rmax > max(bound[:,1]-bound[:,0])/2:
                raise ValueError(
                    'rmax cannot be larger than half the largest dimension in '
                    f'boundary, use rmax < {max(bound[:,1]-bound[:,0])/2}'
                )

    #init function that returns energy from list of pairwise distances
    #if function is given, use that
    if callable(pairpotential):
        pot_fun = pairpotential
    #otherwise use nearest or linearly interpolate
    else:
        #bin edges and centres for pairpotential
        if pairpotential_binedges is None:
            pairpotential_binedges = rvals
        pairpotential_bincenter = \
            (pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2
        
        #convert bool values of interpolate to nearest or linear
        if type(interpolate)==bool:
            if interpolate:
                interpolate='linear'
            else:
                interpolate='nearest'

        from scipy.interpolate import interp1d   
        pot_fun = interp1d(
            pairpotential_bincenter,
            pairpotential,
            kind=interpolate,
            bounds_error=False,
            fill_value='extrapolate'
        )
    
    #define a reduced area for test-particles away from all boundaries
    if avoid_boundary:
        reduced_boundary = boundary.copy()
        reduced_boundary[:,:,0] += rmax
        reduced_boundary[:,:,1] -= rmax
    
    if insert_grid:
        base_n_ins = n_ins
    
    #initialize arrays to store values
    counter = np.zeros(nr)
    pair_correlation = np.zeros(nr)
    
    #loop over all timesteps / independent sets of coordiates
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #check if coords are within boundary
        if (coords<bound[:,0]).any() or (coords>bound[:,1]).any():
            print()#newline
            warn('not all coordinates are within boundary in coordinate set '
                 f'{i}, ignoring spurious coords',RuntimeWarning)
            coords = coords[
                np.logical_and(
                    coords>=bound[:,0],
                    coords<bound[:,1]
                ).all(axis=1)
            ]
        
        #use input coords or generate new test-particle coordinates for each set
        if type(ins_coords) != type(None):
            trialparticles = ins_coords
            n_ins = len(ins_coords)
        elif insert_grid and avoid_boundary:
            trialparticles = _coord_grid_in_box(reduced_boundary[i],n=base_n_ins)
            n_ins = len(trialparticles)
        elif insert_grid:
            trialparticles = _coord_grid_in_box(bound,n=base_n_ins)
            n_ins = len(trialparticles)
        elif avoid_coordinates and avoid_boundary:
            trialparticles = _rand_coord_at_dist(reduced_boundary[i],coords,rmin,n=n_ins)
        elif avoid_coordinates:
            trialparticles = _rand_coord_at_dist(bound,coords,rmin,n=n_ins)
        elif avoid_boundary:
            trialparticles = _rand_coord_in_box(reduced_boundary[i],n=n_ins)
        else:
            trialparticles = _rand_coord_in_box(bound,n=n_ins)
         
        #init KDTree for fast pairfinding
        if periodic_boundary:
            coords = coords - bound[:,0]#shift box to origin
            tree = cKDTree(coords,boxsize=bound[:,1]-bound[:,0])
        else:
            tree = cKDTree(coords)
        
        #set default neighbor_upper_bound
        if type(neighbors_upper_bound)==type(None):
            k = len(coords)
        else:
            k = min([neighbors_upper_bound,len(coords)])
        
        #find all pairs with one particle from testparticles and one from coordinates
        distances,_ = tree.query(trialparticles,k=k,distance_upper_bound=rmax,
                                 workers=workers)
        
        #cKDTree pads rows with np.inf to get correct length, work with mask
        #to only work on finite values
        mask = np.isfinite(distances) & (distances>0)
        
        #calculate total potential energy of insertion (psi) for each testparticle
        #(row) by summing each pairwise potential energy u(r)
        if avoid_boundary:
            exp_psi = np.empty(n_ins)
            for j,(row,msk) in enumerate(zip(distances,mask)):
                exp_psi[j] = np.exp(-np.sum(pot_fun(row[msk])))

        else:
            if periodic_boundary:
                #calculate correction factor for each distance bin to account 
                # for missing information
                boundarycorr = _sphere_shell_vol_frac_in_cuboid_periodic(
                    rvals,
                    min(bound[:,1]-bound[:,0])
                )[np.newaxis,:]
                
            else:
                #calculate correction factor for each testparticle for each dist bin
                # to account for missing information around particles near boundary.
                #boundary is shifted for coordinate system with origin in particle
                boundarycorr = _sphere_shell_vol_frac_in_cuboid(
                    rvals,
                    bound-trialparticles[:,:,np.newaxis]
                )
            #sum pairwise energy per particle per distance bin, then correct
            # each bin for missing volume, then sum and convert to probability 
            # e^(-psi)
            exp_psi = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                dist = row[msk]
                exp_psi[j] = np.histogram(dist,bins=rvals,weights=pot_fun(dist))[0]
            
            #calculate probability of testparticles
            exp_psi = np.exp(-np.sum(exp_psi/boundarycorr,axis=1))
        
        #count how many pairs each particle contributes to each bin
        if _numba_available:
            counts = _apply_hist_nb(distances,mask,rvals)
        else:
            counts = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                counts[j] = np.histogram(row[msk],bins=rvals)[0]
        
        #take average insertion probability of all test-particles
        prob_tot = np.mean(exp_psi)
        
        #boundary correct counts if needed
        if avoid_boundary:
            corrcounts = counts.copy()
        else:
            corrcounts = counts.copy() / boundarycorr
        
        #count up the insertion probabilities counting towards each bin using 
        #the edge-corrected count number so that all distances have equal 
        #weight with respect to the averaging in prob_tot
        prob_r = np.sum(corrcounts*exp_psi[:,np.newaxis],axis=0)
        corrcounts = corrcounts.sum(axis=0)
        prob_r[corrcounts!=0] /= corrcounts[corrcounts!=0]
        
        #add to overall lists weighted by total count
        pair_correlation += counts.sum() * prob_r / prob_tot
        counter += counts.sum(axis=0)
      
    #take total count weighted average of all datasets
    pair_correlation /= counter.sum()
    pair_correlation[counter==0] = 0
    
    return rvals,pair_correlation,counter

def _rdf_insertion_binned_3d_sphere(coordinates,pairpotential,rmin=0,rmax=10,
    dr=None,boundary=None,pairpotential_binedges=None,n_ins=1000,
    interpolate=True,avoid_boundary=False,neighbors_upper_bound=None,workers=1):
    """not intended to be called directly, see top-level `rdf_dist_hist_3d`
    function. This function calculates the rdf from insertion of test-particles
    into sets of existing 3D coordinates in spherical boundaries, averaged over 
    bins of width dr, and based on the pairwise interaction potential u(r) (in 
    units of kT).

    Parameters
    ----------
    coordinates : list-like of numpy.array
        List of sets of coordinates, where each item along the 0th dimension is
        a n*3 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one z-stack, a time step from a 
        video, etc.), with each element of the array of form  `[z,y,x]`. Each 
        set of coordinates is not required to have the same number of particles
        but all coordinates must be within the bounding box(es) given in 
        `boundary`.
    pairpotential : iterable or callable
        list of values for the pairwise interaction potential. Must have length
        of `len(pairpotential_binedges)-1` and be in units of thermal energy 
        kT. Alternatively, any callable (function) which takes only a numpy 
        array of pairwise distances and returns the pairpotential can be given.
    rmin : float, optional
        lower cut-off for the pairwise distance. The default is 0.
    rmax : float
        cut-off radius for the pairwise distance (right edge of last bin).
    dr : float
        bin width of the pairwise distance bins.
    boundary : tuple or list thereof, optional
        origin and radius of the sphere defining the bounding box of the 
        coordinates given as `(z,y,x,radius)` if all coordinate sets share the 
        same boundary, or a list of such tuples of the same length as 
        `coordinates` for specifying boundaries of each set in  `coordinates` 
        separately. The default is the minimum radius sphere centered around
        the mean positions of the particles.
    pairpotential_binedges : iterable, optional
        bin edges corresponding to the values in `pairpotential. The default 
        is None, which uses the bins defined by `rmin`, `rmax` and `dr`. This 
        parameter is ignored if pairpotential is a callable (function)
    n_ins : int, optional
        the number of test-particles to insert into each item in `coordinates`.
        The default is 1000.
    interpolate : bool, optional
        whether to use linear interpolation for calculating the interaction of 
        two particles using the values in `pairpotential`. The default is True.
        If False, the nearest bin value is used. This parameter is ignored if 
        pairpotential is a callable (function).
    avoid_boundary : bool, optional
        if True, all test-particles are inserted at least `rmax` away from any 
        of the surfaces defined in `boundary` to avoid effects of the finite
        volume of the bounding box. The default is False, which uses an 
        analytical correction factor for missing volume of test-particles near 
        the boundaries.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. Note: this is
        ignored when `periodic_boundary=True`. The default is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    pair_correlation : numpy.array
        values for the rdf / pair correlation function in each bin
    counter : numpy.array
        number of pair counts that contributed to the (mean) values in each bin
    """
    #create bin edges and bin centres for r
    coordinates = _check_coordinate_input(coordinates)
    rvals = _get_rvals(rmin, rmax, dr)
    rcent = (rvals[1:]+rvals[:-1])/2
    nr = len(rcent)
    nf = len(coordinates)

    #set default boundary or make sure the length is correct
    if boundary is None:
        boundary = []
        for coord in coordinates:
            bound = np.mean(coord,axis=0)
            rad = np.sqrt(np.sum((coord-bound)**2,axis=1)).max()
            boundary.append(np.concatenate([bound,rad]))
        boundary = np.array(boundary)
    else:
        boundary = np.array(boundary)
        if boundary.ndim == 1:
            boundary = np.broadcast_to(boundary, (nf,4))

    if len(boundary) != nf:
                raise ValueError('lengths of coordinates and boundaries must '
                                 'match')
    if any(rmax >= 2*boundary[:,3]):
        raise ValueError('rmax cannot be larger than 2 times the bounding '
                         f'circle radius, use `rmax<{2*boundary[:,3]}`')

    #init function that returns energy from list of pairwise distances
    #if function is given, use that
    if callable(pairpotential):
        pot_fun = pairpotential
    #otherwise use nearest or linearly interpolate
    else:
        #bin edges and centres for pairpotential
        if pairpotential_binedges is None:
            pairpotential_binedges = rvals
        pairpotential_bincenter = \
            (pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2
        
        #convert bool values of interpolate to nearest or linear
        if type(interpolate)==bool:
            if interpolate:
                interpolate='linear'
            else:
                interpolate='nearest'

        from scipy.interpolate import interp1d   
        pot_fun = interp1d(
            pairpotential_bincenter,
            pairpotential,
            kind=interpolate,
            bounds_error=False,
            fill_value='extrapolate'
        )
    
    #initialize arrays to store values
    counter = np.zeros(nr)
    pair_correlation = np.zeros(nr)
    
    #loop over all timesteps / independent sets of coordiates
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #check if coords in boundary, if not raise warning and remove
        d = np.sqrt(np.sum((bound[:3]-coords)**2,axis=1))
        if any(d>bound[3]):
            print()#newline
            warn('not all coordinates are within boundary in coordinate set '
                 f'{i}, ignoring spurious coords',RuntimeWarning)
            coords = coords[d<=bound[3]]
            d = d[d<=bound[3]]
                
        #generate new test-particle coordinates for each set
        if avoid_boundary:
            trialparticles = _rand_coord_in_sphere(bound[:3],bound[3]-rmax,n=n_ins)
        else:
            trialparticles = _rand_coord_in_sphere(bound[:3],bound[3],n=n_ins)
        
        #init KDTree for fast pairfinding
        tree = cKDTree(coords)
        
        #find all pairs with one particle from testparticles and one from coordinates
        distances,_ = tree.query(trialparticles,k=len(coords),
                                 distance_upper_bound=rmax,workers=workers)
        
        #cKDTree pads rows with np.inf to get correct length, work with masked
        #arrays to only work on finite values
        mask = np.isfinite(distances) & (distances>0)
        distances = np.ma.masked_array(distances,mask)
        
        #calculate total potential energy of insertion (psi) for each testparticle
        #(row) by summing each pairwise potential energy u(r)
        if avoid_boundary:
            exp_psi = np.empty(n_ins)
            for j,(row,msk) in enumerate(zip(distances,mask)):
                exp_psi[j] = np.exp(-np.sum(pot_fun(row[msk])))
        
        else:
            #calculate correction factor for each testparticle for each dist bin
            # to account for missing information around particles near boundary.
            boundarycorr = _sphere_shell_vol_frac_in_sphere(
                rvals,
                np.sqrt(np.sum((trialparticles-bound[:3])**2,axis=1)),
                bound[3]
            )
                
            #sum pairwise energy per particle per distance bin, then correct
            # each bin for missing volume, then sum and convert to probability 
            # e^(-psi)
            exp_psi = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                dist = row[msk]
                exp_psi[j] = np.histogram(dist,bins=rvals,weights=pot_fun(dist))[0]
            
            #calculate probability of testparticles
            exp_psi = np.exp(-np.sum(exp_psi/boundarycorr,axis=1))
            
        #count how many pairs each particle contributes to each bin
        if _numba_available:
            counts = _apply_hist_nb(distances,mask,rvals)
        else:
            counts = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                counts[j] = np.histogram(row[msk],bins=rvals)[0]
        
        #take average insertion probability of all test-particles
        prob_tot = np.mean(exp_psi)
        
        #boundary correct counts if needed
        if avoid_boundary:
            corrcounts = counts.copy()
        else:
            corrcounts = counts.copy() / boundarycorr
        
        #count up the insertion probabilities counting towards each bin using 
        #the edge-corrected count number so that all distances have equal 
        #weight with respect to the averaging in prob_tot
        prob_r = np.sum(corrcounts*exp_psi[:,np.newaxis],axis=0)
        corrcounts = corrcounts.sum(axis=0)
        prob_r[corrcounts!=0] /= corrcounts[corrcounts!=0]
        
        #add to overall lists weighted by total count
        pair_correlation += counts.sum() * prob_r / prob_tot
        counter += counts.sum(axis=0)
      
    #take total count weighted average of all datasets
    pair_correlation[counter!=0] /= counter.sum()
    pair_correlation[counter==0] = 0
    
    return rvals,pair_correlation,counter
    
def _rdf_insertion_binned_3d_custom(coordinates,pairpotential,boundary_func,
        boundary,testparticle_func,rmin=0,rmax=10,dr=None,
        pairpotential_binedges=None,n_ins=1000,interpolate=True,
        avoid_coordinates=False,neighbors_upper_bound=None,workers=1):
    """not intended to be called directly, see top-level `rdf_dist_hist_3d`
    function. This function calculates the rdf from insertion of test-particles
    using boundaries and corrections supplied via arguments so that custom 
    boundary geometries can be corrected for, as well as for a list of datasets
    with different boundary coditions between them, averaged over bins of width
    dr, and based on the pairwise interaction potential u(r) (in units of kT).
    
    Parameters
    ----------
    coordinates : numpy.array or list-like of numpy.array
        list of sets of coordinates, where each item along the 0th dimension is
        a N*3 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one z-stack, a time step from a 
        video, etc.), with each element of the array of form  `[z,y,x]`. Each 
        set of coordinates is not required to have the same number of particles
        but must have the same particle density. When no boundaries or a single
        boundary is given, it is assumed all coordinate sets share these same
        boundaries.
    pairpotential : iterable
        list of values for the pairwise interaction potential. Must have length
        of `len(pairpotential_binedges)-1` and be in units of thermal energy kT
    boundaryfunc : callable or list thereof
        function to correct for finite boundary conditions. Must take three 
        arguments: a numpy array of bin edges, an N×3 numpy array of 
        coordinates and a boundary as specified in `boundary`, and return an 
        N × `len(bin edges)-1` numpy array with a value between 0 and 1 
        specifying the fraction of the volume of each spherical shell which is 
        within the boundary. See e.g. `_sphere_shell_vol_frac_in_cuboid`.
    boundary : tuple or list thereof
        boundary specification as required for boundaryfunc.
    testparticle_func : callable or list thereof
        function that randomly generates testparticles within the boundary. 
        Must take 2 arguments, resp. the boundary and n_ins, or 4 argumants, 
        resp. the boundaries, coordinates, minimum distance and n_ins when
        `avoid_coordinates=True`. See e.g. `_rand_coord_in_box` or 
        `_rand_coord_at_dist`.
    rmin : float, optional
        lower cut-off for the pairwise distance. The default is 0.
    rmax : float
        cut-off radius for the pairwise distance (right edge of last bin).
    dr : float
        bin width of the pairwise distance bins.
    boundary : TYPE, optional
        DESCRIPTION. The default is None.
    pairpotential_binedges : iterable, optional
        bin edges corresponding to the values in `pairpotential. The default 
        is None, which uses the bins defined by `rmin`, `rmax` and `dr`.
    n_ins : int, optional
        the number of test-particles to insert into each item in `coordinates`.
        The default is 1000.
    interpolate : bool, optional
        whether to use linear interpolation for calculating the interaction of 
        two particles using the values in `pairpotential`. The default is True.
        If False, the nearest bin value is used.
    avoid_coordinates : bool, optional
        whether to insert test-particles at least `rmin` away from the center 
        of any of the 'real' coordinates in `coordinates`. The default is 
        False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.
    workers : int, optional
        number of workers to use for parallel processing during the neighbour
        detection step. If -1 is given all CPU threads are used. The default 
        is 1.

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    pair_correlation : numpy.array
        values for the rdf / pair correlation function in each bin
    counter : numpy.array
        number of pair counts that contributed to the (mean) values in each bin
    """

    _check_coordinate_input(coordinates)
    
    #create bin edges and bin centres for r
    rvals = _get_rvals(rmin, rmax, dr)
    rcent = (rvals[1:]+rvals[:-1])/2
    nf = len(coordinates)
    nr = len(rcent)
    
    #check if testparticle_func is given
    if testparticle_func is None:
        raise ValueError('testparticle_func must be given when using custom '
                         'boundary handling')
    
    #make sure boundary is given and is list
    if boundary is None:
        raise ValueError('when using custom boundary handling, `boundary` '
                         'must be given')
    elif type(boundary) != list:
        boundary = [boundary]*nf
    
    #assure lists for functions
    if callable(boundary_func) and callable(testparticle_func):
        boundary_func = [boundary_func]*nf
        testparticle_func = [testparticle_func]*nf
    
    #can't mix list and single func
    elif callable(boundary_func) or callable(testparticle_func):
        raise TypeError('either a single boundary, boundary_func and '
                        'testparticle_func must be given, or all must be a '
                        'list matching the length of coordinates')
    
    #check lengths
    if len(coordinates) != len(boundary):
        raise ValueError('lengths of `boundary` and `coordinates` must match')
    if len(coordinates) != len(boundary_func):
        raise ValueError('lengths of boundary correction functions and '
                         '`coordinates` must match')
    if len(coordinates) != len(testparticle_func):
        raise ValueError('lengths of `testparticle_func` and `coordinates` '
                         'must match')
    
    #init function that returns energy from list of pairwise distances
    #if function is given, use that
    if callable(pairpotential):
        pot_fun = pairpotential
    #otherwise use nearest or linearly interpolate
    else:
        #bin edges and centres for pairpotential
        if pairpotential_binedges is None:
            pairpotential_binedges = rvals
        pairpotential_bincenter = \
            (pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2
        
        #convert bool values of interpolate to nearest or linear
        if type(interpolate)==bool:
            if interpolate:
                interpolate='linear'
            else:
                interpolate='nearest'

        from scipy.interpolate import interp1d   
        pot_fun = interp1d(
            pairpotential_bincenter,
            pairpotential,
            kind=interpolate,
            bounds_error=False,
            fill_value='extrapolate'
        )
    
    #initialize arrays to store values
    counter = np.zeros(nr)
    pair_correlation = np.zeros(nr)
    
    #loop over all timesteps / independent sets of coordiates
    for i,(bound,boundfunc,testfunc,coords) in \
        enumerate(zip(boundary,boundary_func,testparticle_func,coordinates)):
        
        #generate new test-particle coordinates for each set
        if avoid_coordinates:
            trialparticles = testfunc(bound,coords,rmin,n_ins)
        else:
            trialparticles = testfunc(bound,n_ins)
        
        #init KDTree for fast pairfinding
        tree = cKDTree(coords)
        
        #set default neighbor_upper_bound
        if type(neighbors_upper_bound)==type(None):
            k = len(coords)
        else:
            k = min([neighbors_upper_bound,len(coords)])
        
        #find all pairs with one particle from testparticles and one from coordinates
        distances,_ = tree.query(trialparticles,k=k,distance_upper_bound=rmax,
                                 workers=workers)
        
        #cKDTree pads rows with np.inf to get correct length, work with masked
        #arrays to only work on finite values
        mask = np.isfinite(distances) & (distances>0)
        distances = np.ma.masked_array(distances,mask)
        
        #calculate total potential energy of insertion (psi) for each testparticle
        #(row) by summing each pairwise potential energy u(r) 
        #calculate correction factor for each testparticle for each dist bin
        # to account for missing information around particles near boundary.
        #boundary is shifted for coordinate system with origin in particle
        boundarycorr = boundfunc(
            rvals,
            trialparticles,
            bound
        )
                
        #sum pairwise energy per particle per distance bin, then correct
        # each bin for missing volume, then sum and convert to probability 
        # e^(-psi)
        exp_psi = np.empty((n_ins,nr))
        for j,(row,msk) in enumerate(zip(distances,mask)):
            dist = row[msk]
            exp_psi[j] = np.histogram(dist,bins=rvals,weights=pot_fun(dist))[0]
            
        #calculate probability of testparticles
        exp_psi = np.exp(-np.sum(exp_psi/boundarycorr,axis=1))
            
        #count how many pairs each particle contributes to each bin
        if _numba_available:
            counts = _apply_hist_nb(distances,mask,rvals)
        else:
            counts = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                counts[j] = np.histogram(row[msk],bins=rvals)[0]
        
        #take average insertion probability of all test-particles
        prob_tot = np.mean(exp_psi)
        
        #boundary correct counts
        corrcounts = counts.copy() / boundarycorr
        
        #count up the insertion probabilities counting towards each bin using 
        #the edge-corrected count number so that all distances have equal 
        #weight with respect to the averaging in prob_tot
        prob_r = np.sum(corrcounts*exp_psi[:,np.newaxis],axis=0)
        corrcounts = corrcounts.sum(axis=0)
        prob_r[corrcounts!=0] /= corrcounts[corrcounts!=0]
        
        #add to overall lists weighted by total count
        pair_correlation += counts.sum() * prob_r / prob_tot
        counter += counts.sum(axis=0)
      
    #take total count weighted average of all datasets
    pair_correlation /= counter.sum()
    pair_correlation[counter==0] = 0
    
    return rvals,pair_correlation,counter

#%% various smaller helper functions
def _get_rvals(rmin,rmax,dr):
    """checks default dr and returns array of bin edges"""
    if dr is None:
        dr = (rmax-rmin)/20
    return np.arange(rmin,rmax+dr,dr)

def _calc_squared_dist(coordinates,trialparticle,rmax):
    """pure python loop over coordinate pairs, returns pairwise distances"""
    #naive implementation, slow
    distances = []
    for coord in coordinates:
        d = np.sum((coordinates-trialparticle)**2)
        if d <= rmax**2:
            distances.append(d)
    return distances

def _check_coordinate_input(coordinates):
    """checks type and shape of coordinate arrays"""
    #if already a list, assure all items in it are arrays
    if isinstance(coordinates,list):
        if not all([type(coord)==np.ndarray for coord in coordinates]):
            coordinates = [np.array(coord) for coord in coordinates]
    
    #if array, assure list of array
    elif isinstance(coordinates,np.ndarray):
        if not np.can_cast(coordinates[0].dtype,float):
            raise TypeError(
                f"dtype `{coordinates[0].dtype}` of `coordinates` can't be "
                "broadcasted to `float`"
            )
        coordinates = [coordinates]
    else:
        raise TypeError(
            f"dtype `{type(coordinates)}` of `coordinates` not supported, use "
            "a list of numpy.array"
        )
    
    #check dimensionality of the arrays, must be shape n×2 or n×3, e.g. 2D
    if any([coord.ndim != 2 for coord in coordinates]):
        raise TypeError('`coordinates` must be a 2-dimensional '
                        'numpy.ndarray or list thereof')
    
    return coordinates

if _numba_available:
    @nb.njit()
    def _apply_hist_nb(distances,mask,rvals):
        """numba overloaded analogue of np.apply_along_axis(np.histogram(),1)"""
        n,r = len(distances),len(rvals)-1
        binned_array = np.empty((n,r),dtype=np.float64)
        
        for i in range(n):
            dist = distances[i][mask[i]]
            binned_array[i] = np.histogram(dist,bins=rvals)[0]
        
        return binned_array

    @nb.njit()
    def _calc_squared_dist_numba(coordinates,trialparticle,rmax):
        """numba-compiled loop over coordinate pairs, returns pairwise distances"""
        #naive implementation
        distances = []
        for i in range(len(coordinates)):
            d = 0
            for j in range(3):
                d += (coordinates[i,j]-trialparticle[j])**2
            if d <= rmax**2:
                distances.append(d)
        return distances
