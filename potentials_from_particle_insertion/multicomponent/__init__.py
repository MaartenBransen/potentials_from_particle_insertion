"""
This file contains codes for multicomponent compatible versions of functions,
i.e. for more than one 'type' of particle.

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

#%%imports

import numpy as np
from warnings import warn
from scipy.spatial import cKDTree

#internal imports
from ..distribution_functions import _numba_available, _get_rvals

if _numba_available:
    from ..distribution_functions import _apply_hist_nb

from ..geometry import (
    _circle_ring_area_frac_in_rectangle_periodic,
    _sphere_shell_vol_frac_in_cuboid,
    _sphere_shell_vol_frac_in_cuboid_periodic,
    _circle_ring_area_frac_in_rectangle,
)
       
from ..generate_coordinates import (
    _rand_coord_in_box,
    _rand_coord_at_dist,
    _coord_grid_in_box,
    #_rand_coord_in_circle,
    #_rand_coord_in_sphere,
    #_rand_coord_on_sphere,
)

from ..pairpotential_iterator import _regulated_updater

#%% public definitions

def rdf_dist_hist_2d(coordinates,rmin=0,rmax=10,dr=None,boundary=None,
        density=None,combinations=None,periodic_boundary=False,
        handle_edge='rectangle',quiet=False,neighbors_upper_bound=None,
        workers=1):
    """calculates g(r) via a 'conventional' distance histogram method for a 
    set of 2D coordinate sets. Provided for convenience.

    Parameters
    ----------
    coordinates : (list-like of) list-like of numpy.ndarray of floats
        list-like where each item along the 0th dimension is an N×2 
        numpy.ndarray of particle coordinates of one of the components, with 
        each row specifying a particle position in the form [`y`,`x`].
        Alternatively, a list-like of the above may be given where each item is
        an independent set of coordinate components (e.g. a time step from a 
        video), in which case every set must have the same number of components
        but is not required to have the same number of particles and is assumed
        to share the same boundaries when no boundaries or only a single set of
        boundaries are given.
        
        For example, for a binary (two-component) system, `coordinates` would 
        be a list containing two numpy.ndarrays in case of a single dataset, or
        a list with M lists each containing 2 arrays in case of M independent 
        datasets.
    rmin : float, optional
        lower bound for the pairwise distance, left edge of 0th bin. The 
        default is 0.
    rmax : float, optional
        upper bound for the pairwise distance, right edge of last bin. The 
        default is 10.
    dr : float, optional
        bin width for the pairwise distance bins. The default is (rmax-rmin)/20
    handle_edge : one of ['none','rectangle','periodic rectangle'], optional
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
        
        The default is 'rectangle'.
    
    boundary : array-like, optional
        positions of the walls that define the bounding box of the coordinates.
        The format must be given as `((ymin,ymax),(xmin,xmax))` when 
        `handle_edge` is `'none'`, `'rectangle'` or `'periodic rectangle'`. All
        components must share the same boundary.
        
        If all coordinate sets share the same boundary, a single such boundary 
        can be given, otherwise a list of such array-likes of the same length 
        as coordinates can be given for specifying boundaries of each set in 
        `coordinates` separately. The default is `None` which determines the 
        smallest set of boundaries encompassing a set of coordinates.
    density : list of float, optional
        number density of particles in the box for each of the components to 
        use for normalizing the g(r) values. The default is the average density
        based on `coordinates` and `boundary`.
    combinations : list of tuple of int
        list of different combinations of the two components to calculate the 
        g(r) for, where the first element is the integer index of the component
        to use for the central reference particles and the second is the index
        of the component to bincount as neighbouring particles around the 
        reference particles. The default is all possible combinations of 
        n components in the order
        
            [(0,0), (0,1), ..., (0,n), (1,0), (1,1), ..., (n,n)]
            
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
    bincounts : list of numpy.array
        array of values for the bins of the radial distribution function for 
        each item in `combinations` (in that order)
    
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
            dr=dr,boundary=boundary,density=density,combinations=combinations,
            periodic_boundary=False,handle_edge=False,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges in rectangular or square box
    elif handle_edge == 'rectangle':
        return _rdf_dist_hist_2d_rectangle(coordinates,rmin=rmin,rmax=rmax,
            dr=dr,boundary=boundary,density=density,combinations=combinations,
            periodic_boundary=False,handle_edge=True,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges in periodic boundary conditions in rectangle/square box
    elif handle_edge == 'periodic rectangle':
        return _rdf_dist_hist_2d_rectangle(coordinates,rmin=rmin,rmax=rmax,
            dr=dr,boundary=boundary,density=density,combinations=combinations,
            periodic_boundary=True,handle_edge=True,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges in circular box
    elif handle_edge == 'circle':
        raise NotImplementedError('handling circular edges  is currently not '
                                  'supported for multicomponent analysis')
    
    #correct for edges using arbitrary correction funcs
    elif callable(handle_edge) or \
        (type(handle_edge)==list and callable(handle_edge[0])):
        raise NotImplementedError('custom edge handling is currently not '
                                  'supported for multicomponent analysis')
    
    #error for unrecognised options for handle_edge
    else:
        raise ValueError(f'{handle_edge} not a valid option for `handle_edge`')
    
def rdf_dist_hist_3d(coordinates,rmin=0,rmax=10,dr=None,handle_edge='cuboid',
        boundary=None,density=None,combinations=None,quiet=False,
        neighbors_upper_bound=None,workers=1):
    """calculates g(r) via a 'conventional' distance histogram method for a 
    set of 3D coordinate sets. Provided for convenience.

    Parameters
    ----------
    coordinates : (list-like of) list-like of numpy.ndarray of floats
        list-like where each item along the 0th dimension is an N×3 
        numpy.ndarray of particle coordinates of one of the components, with 
        each row specifying a particle position in the form [`z`,`y`,`x`].
        Alternatively, a list-like of the above may be given where each item is
        an independent set of coordinate components (e.g. a time step from a 
        series of z-stacks), in which case every set must have the same number 
        of components but is not required to have the same number of particles 
        and is assumed to share the same boundaries when no boundaries or only 
        a single set of boundaries are given.
        
        For example, for a binary (two-component) system, `coordinates` would 
        be a list containing two numpy.ndarrays in case of a single dataset, or
        a list with M lists each containing 2 arrays in case of M independent 
        datasets.
    rmin : float, optional
        lower bound for the pairwise distance, left edge of 0th bin. The 
        default is 0.
    rmax : float, optional
        upper bound for the pairwise distance, right edge of last bin. The 
        default is 10.
    dr : float, optional
        bin width for the pairwise distance bins. The default is (rmax-rmin)/20
    handle_edge : one of ['none','cuboid','periodic cuboid'], optional
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
        
        The default is 'cuboid'.
    
    boundary : array-like, optional
        positions of the walls that define the bounding box of the coordinates.
        The format must be given as `((zmin,zmax),(ymin,ymax),(xmin,xmax))` 
        when `handle_edge` is `'none'`, `'cuboid'` or `'periodic cuboid'`. All
        components must share the same boundary.
        
        If all coordinate sets share the same boundary a single such boundary 
        can be given, otherwise a list of such array-likes of the same length 
        as coordinates can be given for specifying boundaries of each set in 
        `coordinates` separately. The default is `None` which determines the 
        smallest set of boundaries encompassing a set of coordinates.
    density : list of float, optional
        number density of particles in the box for each of the components to 
        use for normalizing the g(r) values. The default is the average density
        based on `coordinates` and `boundary`.
    combinations : list of tuple of int
        list of different combinations of the two components to calculate the 
        g(r) for, where the first element is the integer index of the component
        to use for the central reference particles and the second is the index
        of the component to bincount as neighbouring particles around the 
        reference particles. The default is all possible combinations of 
        n components in the order
        
            [(0,0), (0,1), ..., (0,n), (1,0), (1,1), ..., (n,n)]
            
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
    bincounts : list of numpy.array
        array of values for the bins of the radial distribution function for 
        each item in `combinations` (in that order)
    
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
            boundary=boundary,density=density,combinations=combinations,
            periodic_boundary=False,handle_edge=False,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges in cuboidal box
    elif handle_edge == 'cuboid':
        return _rdf_dist_hist_3d_cuboid(coordinates,rmin=rmin,rmax=rmax,dr=dr,
            boundary=boundary,density=density,combinations=combinations,
            periodic_boundary=False,handle_edge=True,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges in periodic boundary conditions (cube or cuboid)
    elif handle_edge == 'periodic cuboid':
        return _rdf_dist_hist_3d_cuboid(coordinates,rmin=rmin,rmax=rmax,dr=dr,
            boundary=boundary,density=density,combinations=combinations,
            periodic_boundary=True,handle_edge=True,quiet=quiet,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers)
    
    #correct for edges in spherical box
    elif handle_edge == 'sphere':
        raise NotImplementedError('handling spherical edges  is currently not '
                                  'supported for multicomponent analysis')
    
    #correct for edges using arbitrary correction funcs
    elif callable(handle_edge) or \
        (type(handle_edge)==list and callable(handle_edge[0])):
        raise NotImplementedError('custom edge handling is currently not '
                                  'supported for multicomponent analysis')
    
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
    
    Implementation partly based on ref. [1] but with novel corrections for 
    edge effects based on analytical formulas for periodic and nonperiodic 
    boundary conditions.

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
            based on ref [2].
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
    [1] Stones, A. E., Dullens, R. P. A., & Aarts, D. G. A. L. (2019). Model-
    Free Measurement of the Pair Potential in Colloidal Fluids Using Optical 
    Microscopy. Physical Review Letters, 123(9), 098002. 
    https://doi.org/10.1103/PhysRevLett.123.098002
    
    [2] Markus Seserno (2014). How to calculate a three-dimensional g(r) under
    periodic boundary conditions.
    https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf
    """
    #just copied from single component so far
    raise NotImplementedError('2D multicomponent not yet implemented')
    
    '''
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
            handle_edge,testparticle_func,rmin=rmin,rmax=rmax,dr=dr,
            boundary=boundary,pairpotential_binedges=pairpotential_binedges,
            n_ins=n_ins,interpolate=interpolate,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers,
            **kwargs)
    
    #error for unrecognised options for handle_edge
    else:
        raise ValueError(f'{handle_edge} not a valid option for `handle_edge`')
    '''

def rdf_insertion_binned_3d(coordinates,pairpotential,rmin=0,rmax=10,dr=None,
        handle_edge='cuboid',boundary=None,pairpotential_binedges=None,
        n_ins=1000,interpolate=True,avoid_boundary=False,
        avoid_coordinates=False,neighbors_upper_bound=None,workers=1,
        testparticle_func=None,**kwargs):
    """Calculate g(r) from insertion of test-particles into sets of existing
    2D coordinates, averaged over bins of width dr, and based on the pairwise 
    interaction potential u(r) (in units of kT).
    
    Implementation partly based on ref. [1] but with novel corrections for 
    edge effects based on analytical formulas for periodic and nonperiodic 
    boundary conditions.

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
            ref. [2]
        *   `'periodic' cuboid'`: like `'cuboid'`, except in periodic 
            boundary conditions (i.e. one side wraps around to the other). 
            Based on ref. [3].
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
    [1] Stones, A. E., Dullens, R. P. A., & Aarts, D. G. A. L. (2019). Model-
    Free Measurement of the Pair Potential in Colloidal Fluids Using Optical 
    Microscopy. Physical Review Letters, 123(9), 098002. 
    https://doi.org/10.1103/PhysRevLett.123.098002
    
    [2] Kopera, B. A. F., & Retsch, M. (2018). Computing the 3D Radial 
    Distribution Function from Particle Positions: An Advanced Analytic 
    Approach. Analytical Chemistry, 90(23), 13909–13914. 
    https://doi.org/10.1021/acs.analchem.8b03157
    
    [3] Markus Seserno (2014). How to calculate a three-dimensional g(r) under
    periodic boundary conditions.
    https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf
    """
    #check the type of edge handling, and pass off to appropriate subfunction:
    #don't correct for edge effects, but assume rectangular box otherwise
    if handle_edge is None or handle_edge is False or handle_edge == 'none':
        return _rdf_insertion_binned_3d_cuboid(coordinates, pairpotential, 
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
    elif handle_edge == 'sphere' or callable(handle_edge) or \
        (type(handle_edge)==list and callable(handle_edge[0])):
        raise NotImplementedError('TPI with sphere or callable not implemented')
    
    #error for unrecognised options for handle_edge
    else:
        raise ValueError(f'{handle_edge} not a valid option for `handle_edge`')
    '''
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
            handle_edge,testparticle_func,rmin=rmin,rmax=rmax,dr=dr,
            boundary=boundary,pairpotential_binedges=pairpotential_binedges,
            n_ins=n_ins,interpolate=interpolate,avoid_boundary=avoid_boundary,
            neighbors_upper_bound=neighbors_upper_bound,workers=workers,
            **kwargs)
    '''
    

#defs
def run_iteration(coordinates,pair_correlation_func,initial_guess=None,rmin=0,
                  rmax=10,dr=None,convergence_tol=1e-5,max_iterations=100,
                  zero_clip=1e-20,regulate=False,**kwargs):
    """
    Run the algorithm to solve for the pairwise potential that most accurately
    reproduces the radial distribution function using test-particle insertion,
    as described in ref. [1]. 

    Parameters
    ----------
    coordinates : list-like of numpy.array
        List of sets of coordinates, where each item along the 0th dimension is
        a n*3 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one z-stack, a time step from a 
        video, etc.), with each element of the array of form  `[z,y,x]` or 
        `[y,x]` in case of 2D data. Each set of coordinates is not required to 
        have the same number of particles but all stacks must share the same 
        bounding box as given by `boundary`, and all coordinates must be within
        this bounding box.
    pair_correlation_func : list of float
        bin values for the true pair correlation function that the algorithm 
        will try to match iteratively.
    boundary : array-like of form `([(zmin,zmax),](ymin,ymax),(xmin,xmax))`
        positions of the walls that define the bounding box of the coordinates.
        Number of dimensions must match coordinates.
    initial_guess : list of float, optional
        Initial guess for the particle potential on the 0th iteration. The 
        default is None which gives 0 in each bin.
    rmin : float, optional
        left edge of the smallest bin in interparticle distance r to consider.
        The default is 0.
    rmax : float, optional
        Right edge of the largest bin in interparticle distance r to consider.
        The default is 20.
    dr : float, optional
        Stepsize or bin width in interparticle distance r. The default is 0.5.
    convergence_tol : float, optional
        target value for χ², if it dips below this value the iteration is 
        considered to be converged and ended. The default is `1e-5`.
    max_iterations : int, optional
        Maximum number of iterations after which the algorithm is ended. The
        default is 100.
    zero_clip : float, optional
        values below the value of zero-clip are set to this value to avoid
        devision by zero errors. The default is `1e-20`.
    regulate : bool, optional
        if True, use regularization to more gently nudge towards the input g(r)
        at the cost of slower convergence. Experimental option. The default is
        `False`.
    **kwargs : key=value
        Additional keyword arguments are passed on to `rdf_insertion_binned_2d`
        or `rdf_insertion_binned_3d`

    Returns
    -------
    χ² : list of float
        summed squared error in the pair correlation function for each 
        iteration
    pairpotential : list of list of float
        the values for the pair potential in each bin for each iteration
    paircorrelation : list of list of float
        the values for the pair correlation function from test-particle
        insertion for each iteration
    counts : list of list of int
        number of pair counts contributing to each bin in each iteration
    
    References
    ----------
    [1] Stones, A. E., Dullens, R. P. A., & Aarts, D. G. A. L. (2019). Model-
    Free Measurement of the Pair Potential in Colloidal Fluids Using Optical 
    Microscopy. Physical Review Letters, 123(9), 098002. 
    https://doi.org/10.1103/PhysRevLett.123.098002
    
    See also
    --------
    rdf_insertion_binned_2d : 2D routine for g(r) from test-particle insertion
    rdf_insertion_binned_3d : 3D routine for g(r) from test-particle insertion
    """
    
    #check coords, assure arrays
    coordinates,_ = _check_multicomponent_coordinate_input(coordinates)
    pair_correlation_func = np.array(pair_correlation_func)
    
    #create values for bin edges and centres of r
    rvals = _get_rvals(rmin,rmax,dr)
    rcent = (rvals[1:]+rvals[:-1])/2
    nr = len(rcent)
    nc = len(coordinates[0])
    
    #create all combinations of components
    combinations = _get_combinations(nc)
    ncomb = len(combinations)
    
    #check inputs
    if len(pair_correlation_func) != ncomb:
        raise ValueError(f'`pair_correlation_func` must have length {ncomb} '
                         f'for {nc} components')
    if initial_guess is None:
        initial_guess = np.zeros((ncomb,nr))
    elif len(initial_guess) != ncomb:
        raise ValueError(f'`initial_guess` must have length {ncomb} '
                         f'for {nc} components')
    if any([len(p) != len(rcent) for p in pair_correlation_func]):
        raise ValueError('length of elements in `pair_correlation_func` does '
                         'not match rmax and dr')
    
    
    
    #check dimensionality, select appropriate rdf_insertion routine
    dims = np.shape(coordinates[0][0])[1]

    if dims == 2:
        rdf_insertion_binned = rdf_insertion_binned_2d
    elif dims == 3:
        rdf_insertion_binned = rdf_insertion_binned_3d
    else:
        raise ValueError('array(s) in `coordinates` must have 2 or 3 columns')
      
    #set up for 0th iteration seperately
    pairpotential = [initial_guess]
    _,newpaircorrelation,c,_ = rdf_insertion_binned(
        coordinates,
        initial_guess,
        rmin=rmin,
        rmax=rmax,
        dr=dr,
        **kwargs
    )
    #avoid dividing by zero
    pair_correlation_func[pair_correlation_func<zero_clip]=zero_clip
    newpaircorrelation[newpaircorrelation<zero_clip] = zero_clip
    paircorrelation = [newpaircorrelation]
    chi_squared = [np.mean((newpaircorrelation - pair_correlation_func)**2)]
    print('\riteration 0, χ²={:4g}'.format(chi_squared[-1]))
    
    #start the main iterative loop
    i = 1
    counters = [c]
    while i < max_iterations:
        
        #calculate the new pairwise potential, with relaxation
        if regulate:
            newpotential = _regulated_updater(
                np.exp(-pairpotential[-1]),
                np.exp(-pairpotential[-1])*pair_correlation_func/newpaircorrelation,
                i
                )
        else:
             newpotential = np.exp(-pairpotential[-1])*pair_correlation_func/newpaircorrelation   

        newpotential[newpotential<zero_clip] = zero_clip
        newpotential = -np.log(newpotential)
        pairpotential.append(newpotential)
        
        #use it to calculate the new g(r)
        _,newpaircorrelation,c,_ = rdf_insertion_binned(
            coordinates,
            newpotential,
            rmin=rmin,
            rmax=rmax,
            dr=dr,
            **kwargs
        )
        counters.append(c)
        newpaircorrelation[newpaircorrelation<zero_clip] = zero_clip
        #newpaircorrelation[newpaircorrelation>20] = 20
        paircorrelation.append(newpaircorrelation)
        
        #calculate summed squared error
        chi_squared.append(np.mean((newpaircorrelation - pair_correlation_func)**2))
        
        print('\riteration {:}, χ²={:4g}'.format(i,chi_squared[-1]))
        if chi_squared[-1] < convergence_tol:
            break
        
        i += 1
    
    return chi_squared,pairpotential,paircorrelation,counters,combinations


#%% private dist hist definitions

def _rdf_dist_hist_2d_rectangle(coordinates,rmin=0,rmax=10,dr=None,
        boundary=None,density=None,combinations=None,periodic_boundary=False,
        handle_edge=True,quiet=False,neighbors_upper_bound=None,workers=1):
    """calculates g(r) via a 'conventional' distance histogram method for a 
    different combinations out of multicomponent (e.g. binary, ternary, ...) 
    2D coordinate sets. Edge correction is fully analytical for both periodic 
    and nonperiodic boundary conditions.

    Parameters
    ----------
    coordinates : (list-like of) list-like of numpy.ndarray of floats
        list-like where each item along the 0th dimension is an N×2 
        numpy.ndarray of particle coordinates of one of the components, with 
        each row specifying a particle position in the form [`y`,`x`].
        Alternatively, a list-like of the above may be given where each item is
        an independent set of coordinate components (e.g. a time step from a 
        video), in which case every set must have the same number of components
        but is not required to have the same number of particles and is assumed
        to share the same boundaries when no boundaries or only a single set of
        boundaries are given.
        
        For example, for a binary (two-component) system, `coordinates` would 
        be a list containing two numpy.ndarrays in case of a single dataset, or
        a list with M lists each containing 2 arrays in case of M independent 
        datasets.
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
    density : list of float, optional
        number density of particles in the box for each of the components to 
        use for normalizing the g(r) values. The default is the average density
        based on `coordinates` and `boundary`.
    combinations : list of tuple of int
        list of different combinations of the two components to calculate the 
        g(r) for, where the first element is the integer index of the component
        to use for the central reference particles and the second is the index
        of the component to bincount as neighbouring particles around the 
        reference particles. The default is all possible combinations of 
        n components in the order
        
            [(0,0), (0,1), ..., (0,n), (1,0), (1,1), ..., (n,n)]
            
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
    bincounts : list of numpy.array
        array of values for the bins of the radial distribution function for 
        each item in `combinations` (in that order)
    """
    #create bins
    rvals = _get_rvals(rmin, rmax, dr)
    coordinates,multistep = _check_multicomponent_coordinate_input(coordinates)
    nf = len(coordinates)
    nc = len(coordinates[0])
    
    #set default boundary as min and max values in dataset
    if boundary is None:
        boundary = np.array([[
                [min([c[:,0].min() for c in coord]),
                 max([c[:,0].max() for c in coord])],
                [min([c[:,1].min() for c in coord]),
                 max([c[:,1].max() for c in coord])]
            ] for coord in coordinates])
    else:
        boundary = np.array(boundary)
        if boundary.ndim == 2:
            boundary = np.broadcast_to(boundary, (nf,2,2))
    
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
    
    #remove coords outside of boundary
    for i,coords in enumerate(coordinates):
        for j,c in enumerate(coords):
            if (c < bound[:,0]).any() or (c >= bound[:,1]).any():
                warn('not all coordinates are within boundary in '
                     f'coordinate set {i}, ignoring spurious coords',
                     RuntimeWarning)
                coordinates[i][j] = c[
                    np.logical_and(c >= bound[:,0], c < bound[:,1]).all(axis=1)
                ]
    
    #set density to mean number density in dataset
    if not density:
        vol = np.product(boundary[:,:,1]-boundary[:,:,0],axis=1)
        density = np.mean([len(coords)/v for v,coords in zip(vol,coordinates)])
    
    #create all combinations of components if not given
    if combinations is None:
        combinations = [(c0,c1) for c0 in range(nc) for c1 in range(nc)]
    
    #loop over all sets of coordinates
    bincounts = np.zeros((nc**2,len(rvals)-1))
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #print progress
        if not quiet:
            if multistep:
                t = f'\rcalculating distance histogram g(r) {i+1} of {nf}'
            else:
                t = '\rcalculating distance histogram g(r)'
            print(t, end='')
        
        #loop over all combinations of components
        # c0 central reference particles, #c1 the neighbours we are counting
        for j,(c0,c1) in enumerate(combinations):
            
            if periodic_boundary:
                #set up KDTree for fast neighbour finding
                #shift box boundary corner to origin for periodic KDTree
                tree0 = cKDTree(coords[c0]-bound[:,0],
                                boxsize=bound[:,1]-bound[:,0])
                tree1 = cKDTree(coords[c1]-bound[:,0],
                                boxsize=bound[:,1]-bound[:,0])
                
                #count number of particle pairs per bin directly
                counts = tree1.count_neighbors(tree0,rvals,
                                               cumulative=False)[1:]
                
                #optionally apply edge correction for distances beyond boxsize/2
                if handle_edge:
                    
                    boundarycorr = _circle_ring_area_frac_in_rectangle_periodic(
                        rvals,
                        bound[0,1]-bound[0,0]
                    )
                    counts = counts/boundarycorr
                
            #if not periodic_boundary
            else:
                #set up KDTree for fast neighbour finding
                tree1 = cKDTree(coords[c1])
                
                if handle_edge:
                    
                    #set default neighbor_upper_bound
                    if type(neighbors_upper_bound)==type(None):
                        k = len(coords[c1])
                    else:
                        k = min([neighbors_upper_bound,len(coords[c1])])
                    
                    #query tree for any neighbours up to rmax for each particle
                    #separately
                    dist,indices = tree1.query(coords[c0],k=k,
                                              distance_upper_bound=rmax,
                                              workers=workers)
                    
                    #remove pairs with self if needed
                    if c0 == c1:
                        dist = dist[:,1:]
                    
                    #remove padded (infinite) values and anything below rmin
                    mask = np.isfinite(dist) & (dist>=rmin)
            
                    #when dealing with edges, histogram the distances per 
                    #reference particle and apply correction factor for missing
                    #volume to each particle (each row)
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
            
                #otherwise calculate neighbours per bin directly
                else:
                    tree0 = cKDTree(coords[c0])
                    counts = tree1.count_neighbors(tree0,rvals,cumulative=False)[1:]
            
            #normalize and add to overall list
            #counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
            bincounts[j] += counts / (np.pi * (rvals[1:]**2 - rvals[:-1]**2))\
                             / (density[c1]*len(coords[c0]))
        
    #newline
    if not quiet:
        print()
    
    #average all datasets
    bincounts = [b/nf for b in bincounts]
    
    return rvals,tuple(bincounts)


def _rdf_dist_hist_3d_cuboid(coordinates,rmin=0,rmax=10,dr=None,boundary=None,
        density=None,combinations=None,periodic_boundary=False,
        handle_edge=True,quiet=False,neighbors_upper_bound=None,workers=1):
    """calculates g(r) via a 'conventional' distance histogram method for 
    different combinations out of multicomponent (e.g. binary, ternary, ...) 
    2D coordinate sets. Edge correction is fully analytical for both periodic 
    and nonperiodic boundary conditions. Edge correction based on refs [1] and
    [2].

    Parameters
    ----------
    coordinates : (list-like of) list-like of numpy.ndarray of floats
        list-like where each item along the 0th dimension is an N×3 
        numpy.ndarray of particle coordinates of one of the components, with 
        each row specifying a particle position in the form [`z`,`y`,`x`].
        Alternatively, a list-like of the above may be given where each item is
        an independent set of coordinate components (e.g. a time step from a 
        series of z-stacks), in which case every set must have the same number 
        of components but is not required to have the same number of particles 
        and is assumed to share the same boundaries when no boundaries or only 
        a single set of boundaries are given.
        
        For example, for a binary (two-component) system, `coordinates` would 
        be a list containing two numpy.ndarrays in case of a single dataset, or
        a list with M lists each containing 2 arrays in case of M independent 
        datasets.
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
    density : list of float, optional
        number density of particles in the box for each of the components to 
        use for normalizing the g(r) values. The default is the average density
        based on `coordinates` and `boundary`.
    combinations : list of tuple of int
        list of different combinations of the two components to calculate the 
        g(r) for, where the first element is the integer index of the component
        to use for the central reference particles and the second is the index
        of the component to bincount as neighbouring particles around the 
        reference particles. The default is all possible combinations of 
        n components in the order
        
            [(0,0), (0,1), ..., (0,n), (1,0), (1,1), ..., (n,n)]
            
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
    bincounts : list of numpy.array
        array of values for the bins of the radial distribution function for 
        each item in `combinations` (in that order)
    
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
    coordinates,multistep = _check_multicomponent_coordinate_input(coordinates)
    nf = len(coordinates)
    nc = len(coordinates[0])
    
    
    #set default boundary as min and max values in dataset
    if boundary is None:
        boundary = np.array([[
                [min([c[:,0].min() for c in coord]),
                 max([c[:,0].max() for c in coord])],
                [min([c[:,1].min() for c in coord]),
                 max([c[:,1].max() for c in coord])],
                [min([c[:,2].min() for c in coord]),
                 max([c[:,2].max() for c in coord])]
            ] for coord in coordinates])
    else:
        boundary = np.array(boundary)
        if boundary.ndim == 2:
            boundary = np.broadcast_to(boundary, (nf,3,2))
    
    #check rmax and boundary for edge-handling in periodic boundary conditions
    if periodic_boundary:
        for bound in boundary:
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
                    'periodic_boundary=True is only implemented for cubic '
                    'boundaries'
                )
    
    #check rmax and boundary for edge handling without periodic boundaries
    else:
        for bound in boundary:
            if rmax > max(bound[:,1]-bound[:,0]):
                raise ValueError(
                    'rmax cannot be larger than half the largest dimension in '
                    f'boundary, use rmax < {max(bound[:,1]-bound[:,0])/2}'
                )
    
    #remove coords outside of boundary
    for i,(coords,bound) in enumerate(zip(coordinates,boundary)):
        for j,c in enumerate(coords):
            if (c < bound[:,0]).any() or (c >= bound[:,1]).any():
                warn('not all coordinates are within boundary in '
                     f'coordinate set {i}, component {j}. Ignoring spurious '
                     'coords',RuntimeWarning)
                coordinates[i][j] = c[
                    np.logical_and(c >= bound[:,0], c < bound[:,1]).all(axis=1)
                ]
    
    #set density to mean number density in dataset
    if density is None:
        vol = np.product(boundary[:,:,1]-boundary[:,:,0],axis=1)
        density = [
            np.mean([len(coords[c])/v for v,coords in zip(vol,coordinates)]) \
                for c in range(nc)]
    
    #create all combinations of components if not given
    if combinations is None:
        combinations = _get_combinations(nc)
    ncomb = len(combinations)
    
    #loop over all sets of coordinates
    bincounts = np.zeros((ncomb,len(rvals)-1))
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #print progress
        if not quiet:
            if multistep:
                t = f'\rcalculating distance histogram g(r) {i+1} of {nf}'
            else:
                t = '\rcalculating distance histogram g(r)'
            print(t, end='')
        
        #loop over all combinations of components
        # c0 central reference particles, #c1 the neighbours we are counting
        for j,(c0,c1) in enumerate(combinations):
            if periodic_boundary:
                #set up KDTree for fast neighbour finding
                #shift box boundary corner to origin for periodic KDTree
                tree0 = cKDTree(coords[c0]-bound[:,0],
                                boxsize=bound[:,1]-bound[:,0])
                tree1 = cKDTree(coords[c1]-bound[:,0],
                                boxsize=bound[:,1]-bound[:,0])
                
                #count number of particle pairs per bin directly
                counts = tree1.count_neighbors(tree0,rvals,
                                               cumulative=False)[1:]
                
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
                tree1 = cKDTree(coords[c1])
                
                if handle_edge:
                    
                    #set default neighbor_upper_bound
                    if type(neighbors_upper_bound)==type(None):
                        k = len(coords[c1])
                    else:
                        k = min([neighbors_upper_bound,len(coords[c1])])
                    
                    #query tree for any neighbours up to rmax for each particle
                    #separately
                    dist,indices = tree1.query(coords[c0],k=k,
                                              distance_upper_bound=rmax,
                                              workers=workers)
                    
                    #remove pairs with self if needed
                    if c0 == c1:
                        dist = dist[:,1:]
                    
                    #remove padded (infinite) values and anything below rmin
                    mask = np.isfinite(dist) & (dist>=rmin)
            
                    #when dealing with edges, histogram the distances per 
                    #reference particle and apply correction factor for missing
                    #volume to each particle (each row)
                    if _numba_available:
                        counts = _apply_hist_nb(dist,mask,rvals)
                    else:
                        counts = np.zeros((len(dist),len(rvals)-1))
                        for j,(row,msk) in enumerate(zip(dist,mask)):
                            counts[j] = np.histogram(row[msk],bins=rvals)[0]
    
                    boundarycorr=_sphere_shell_vol_frac_in_cuboid(
                        rvals,
                        bound-coords[c0][:,:,np.newaxis]
                        )
                    counts = np.sum(counts/boundarycorr,axis=0)
            
                #otherwise calculate neighbours per bin directly
                else:
                    tree0 = cKDTree(coords[c0])
                    counts = tree1.count_neighbors(tree0,rvals,cumulative=False)[1:]
            
            #normalize and add to overall list
            #counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
            bincounts[j] += counts / (4/3*np.pi * (rvals[1:]**3 - rvals[:-1]**3))\
                             / (density[c1]*len(coords[c0]))
        
    #newline
    if not quiet:
        print()
    
    #average all datasets
    bincounts = [b/nf for b in bincounts]
    
    return rvals,tuple(bincounts),tuple(combinations)

#%% private testparticle definitions
    
def _rdf_insertion_binned_3d_cuboid(coordinates,pairpotential,rmin=0,rmax=10,
    dr=None,boundary=None,pairpotential_binedges=None,n_ins=1000,
    interpolate=True,periodic_boundary=False,ins_coords=None,insert_grid=False,
    avoid_boundary=False,avoid_coordinates=False,neighbors_upper_bound=None,
    workers=1):
    """Calculate g(r) from insertion of test-particles into sets of existing
    3D coordinates, averaged over bins of width dr, and based on the pairwise 
    interaction potential u(r) (in units of kT).
    
    Implementation partly based on ref. [1] but with novel corrections for 
    edge effects based on analytical formulas from refs. [2] and [3] for 
    periodic and nonperiodic boundary conditions respectively.

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
    rmin : float, optional
        lower cut-off for the pairwise distance. The default is 0.
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
    
    [2] Markus Seserno (2014). How to calculate a three-dimensional g(r) under
    periodic boundary conditions.
    https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf
    
    [3] Kopera, B. A. F., & Retsch, M. (2018). Computing the 3D Radial 
    Distribution Function from Particle Positions: An Advanced Analytic 
    Approach. Analytical Chemistry, 90(23), 13909–13914. 
    https://doi.org/10.1021/acs.analchem.8b03157
    """
    #create bin edges and bin centres for r
    rvals = _get_rvals(rmin, rmax, dr)
    rcent = (rvals[1:]+rvals[:-1])/2
    coordinates,multistep = _check_multicomponent_coordinate_input(coordinates)
    nf = len(coordinates)
    nc = len(coordinates[0])
    nr = len(rcent)
    
    #set default boundary as min and max values in dataset
    if boundary is None:
        boundary = np.array([[
                [min([c[:,0].min() for c in coord]),
                 max([c[:,0].max() for c in coord])],
                [min([c[:,1].min() for c in coord]),
                 max([c[:,1].max() for c in coord])],
                [min([c[:,2].min() for c in coord]),
                 max([c[:,2].max() for c in coord])]
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

    if pairpotential_binedges is None:
        pairpotential_binedges = rvals
    pairpotential_bincenter = (pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2

    #convert bool values of interpolate to nearest (no interpolation) and linear
    if type(interpolate)==bool:
        if interpolate:
            interpolate='linear'
        else:
            interpolate='nearest'

    #init function that returns energy from list of pairwise distances
    #if function is given, use that
    pot_fun = []
    for p in pairpotential:
        if callable(p):#if callable use it directly
            pot_fun.append(p)
        else:#interpolate pair potential between points
            from scipy.interpolate import interp1d
            pot_fun.append(interp1d(
                pairpotential_bincenter,
                p,
                kind=interpolate,
                bounds_error=False,
                fill_value='extrapolate'
            ))
    
    #define a reduced area for test-particles away from all boundaries
    if avoid_boundary:
        reduced_boundary = boundary.copy()
        reduced_boundary[:,:,0] += rmax
        reduced_boundary[:,:,1] -= rmax
    
    if insert_grid:
        base_n_ins = n_ins
    
    #create all combinations of components if not given
    combinations = _get_combinations(nc)
    ncomb = len(combinations)
    
    #initialize arrays to store values
    counter = np.zeros((ncomb,nr))
    pair_correlation = np.zeros((ncomb,nr))
    
    #loop over all timesteps / independent sets of coordiates
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #check if coords are within boundary
        for j,coord in enumerate(coords):
            if (coord<bound[:,0]).any() or (coord>bound[:,1]).any():
                print()#newline
                warn('not all coordinates are within boundary in coordinate '
                     f'set {i}, component {j}, ignoring spurious coords',
                     RuntimeWarning)
                coords[j] = coord[
                    np.logical_and(
                        coord>=bound[:,0],
                        coord<bound[:,1]
                    ).all(axis=1)
                ]
        
        #use input coords or generate new test-particle coordinates for each set
        if not ins_coords is None:
            testparticles = ins_coords
            n_ins = len(ins_coords)
        elif insert_grid and avoid_boundary:
            testparticles = _coord_grid_in_box(reduced_boundary[i],
                                               n=base_n_ins)
            n_ins = len(testparticles)
        elif insert_grid:
            testparticles = _coord_grid_in_box(bound,n=base_n_ins)
            n_ins = len(testparticles)
        elif avoid_coordinates and avoid_boundary:
            testparticles = _rand_coord_at_dist(reduced_boundary[i],
                                        np.concatenate(coords),rmin,n=n_ins)
        elif avoid_coordinates:
            testparticles = _rand_coord_at_dist(bound,np.concatenate(coords),
                                                rmin,n=n_ins)
        elif avoid_boundary:
            testparticles = _rand_coord_in_box(reduced_boundary[i],n=n_ins)
        else:
            testparticles = _rand_coord_in_box(bound,n=n_ins)
        
        #calculate boundary correction for finite boundaries only once for each
        #test particle, since we use same testparticles for all combinations
        if not avoid_boundary:
            #for periodic boundary, boundarycorrect per bin
            if periodic_boundary:
                boundarycorr = _sphere_shell_vol_frac_in_cuboid_periodic(
                    rvals,
                    min(bound[:,1]-bound[:,0])
                )[np.newaxis,:]
                testparticles -= bound[:,0]#shift box corner to origin
            #for nonperiodic correct on a per testparticle per bin basis
            else:
                boundarycorr = _sphere_shell_vol_frac_in_cuboid(
                    rvals,
                    bound-testparticles[:,:,np.newaxis]
                )
        
        #init lists and run some routines once per component j
        distances,masks,compcounts,corrcounts = [],[],[],[]
        for j in range(nc):
            
            #init KDTree for fast pairfinding with j'th component as neighbour
            if periodic_boundary:
                tree = cKDTree(coords[j]-bound[:,0],
                               boxsize=bound[:,1]-bound[:,0])
            else:
                tree = cKDTree(coords[j])
            
            #determine neigbour upper bounds
            if neighbors_upper_bound is None:
                nub = len(coords[j])
            else:
                nub = min([neighbors_upper_bound,len(coords[j])])
            
            #calculate distances from testparticles to every component
            dist,_ = tree.query(testparticles,k=nub,distance_upper_bound=rmax,
                                workers=workers)
            mask = np.isfinite(dist) & (dist>0)
            distances.append(dist)
            masks.append(mask)
        
            #count how many pairs each testparticle contributes to each bin
            if _numba_available:
                counts = _apply_hist_nb(dist,mask,rvals)
            else:
                counts = np.empty((n_ins,nr))
                for k,(row,msk) in enumerate(zip(dist,mask)):
                    counts[k] = np.histogram(row[msk],bins=rvals)[0]
            
            #boundary correct counts if needed
            if avoid_boundary:
                corrcounts.append(counts.copy())
            else:
                corrcounts.append(counts.copy() / boundarycorr)
            
            #add uncorrected counts per bin to overall counts
            compcounts.append(counts.sum(axis=0))
        
        
        #init list for total potentials and counts per bin
        psis = np.zeros((nc,n_ins))
        
        #sum the interactions of each c0 with all part of all different compnts
        #c0 is reference component (testparticle), c1 is neighbours (real part)
        for j,(c0,c1) in enumerate(combinations):
            
            dist = distances[c1]
            mask = masks[c1]
            
            #sum on a per component rowwise (per testparticle) basis
            #when avoiding boundaries and add to total for central particle of
            #component c0, to get sum of c0 with all neighbours of all c1's
            if avoid_boundary:
                for k,(row,msk) in enumerate(zip(dist,mask)):
                    psis[c0,k] += np.sum(pot_fun[j](row[msk]))
        
            #otherwise sum each row per bin first, then boundary correct, then 
            #sum all bins for total (corrected) potential energy, add this to
            #overall total for reference particle of c0
            else:
                for k,(row,msk,corr) in enumerate(zip(dist,mask,boundarycorr)):
                    psis[c0,k] += np.sum(
                        np.histogram(
                            row[msk],
                            bins=rvals,
                            weights=pot_fun[j](row[msk])
                        )[0] / corr
                    )
            
        #calculate insertion probabilities from psi's
        prob = np.exp(-psis)
        
        #now loop over combinations again, and calculate weighted average local
        #and ensemble probabilities
        for j,(c0,c1) in enumerate(combinations):
            
            #sum the insertion probabilities counting towards each bin using 
            #the edge-corrected count number so that all distances have equal 
            #weight with respect to the averaging in prob_tot
            prob_r = np.sum(corrcounts[c1]*prob[c0,:,np.newaxis],axis=0)
            corrcount = corrcounts[c1].sum(axis=0)
            prob_r[corrcount!=0] /= corrcount[corrcount!=0]
        
            #divide by ensemble average and add to overall lists
            pair_correlation[j] += compcounts[c1].sum() * prob_r / np.mean(prob[c0])
            counter[j] += compcounts[c1]
            
    #take weighted average of all datasets
    pair_correlation /= counter.sum(axis=1)[:,np.newaxis]
    pair_correlation[counter==0] = 0
    
    return rvals,pair_correlation,counter,combinations


#%% private helper functions

def _listlike(var):
    """returns True if var is list-like, for np.ndarray allows only 1d object 
    array"""
    if type(var) in [list,tuple]:
        return True
    if type(var) == np.ndarray:
        if var.ndim == 1 and var.dtype == object:
            return True
    return False
    

def _check_multicomponent_coordinate_input(coordinates):
    """checks inputs on whether it is list or list of list like, 
    and always returns the latter and a flag indicating which it was"""
    #check outer type, must be at least list of np array
    if not _listlike(coordinates):
        raise TypeError('`coordinates` must be of type `list`, `tuple` or a '
                        '1d numpy.ndarray with `dtype=object`')
    
    #check if single or multiple independent sets
    if _listlike(coordinates[0]):
        if len(set([len(c) for c in coordinates])) != 1:
            raise ValueError('number of components must remain constant in all'
                             'of `coordinates`')
        multistep = True
            
    #if list of coordinate arrays, assure list of lists of arrays
    elif type(coordinates[0]) == np.ndarray:
        coordinates = [coordinates]
        multistep = False
    else:
        raise TypeError('`coordinates` must be list of numpy.array or list of '
                        'list of numpy.array')
    
    if len(coordinates[0])==1:
        warn('dataset containing only single-component data is given')
    
    return coordinates,multistep

def _get_combinations(nc):
    """takes int number of components and returns list of 2-tuples for each 
    pairwise combinations of components"""
    return [(c0,c1) for c0 in range(nc) for c1 in range(c0+1)]
    #return [(c0,c1) for c0 in range(nc) for c1 in range(nc)]
    