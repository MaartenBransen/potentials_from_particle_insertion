#%%imports

import numpy as np
from warnings import warn
from scipy.spatial import cKDTree
from ..distribution_functions import \
    _numba_available,_get_rvals
if _numba_available:
    from ..distribution_functions import _apply_hist_nb
from ..geometry import \
    _circle_ring_area_frac_in_rectangle_periodic,\
    _sphere_shell_vol_frac_in_cuboid,\
    _sphere_shell_vol_frac_in_cuboid_periodic,\
    _circle_ring_area_frac_in_rectangle
        

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
    components : list of tuple of int
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
        testparticle_func=None,combinations=None,**kwargs):
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
    components : list of tuple of int
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
    components : list of tuple of int
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
        combinations = [(c0,c1) for c0 in range(nc) for c1 in range(nc)]
    
    #loop over all sets of coordinates
    bincounts = np.zeros((len(combinations),len(rvals)-1))
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
    
    return rvals,tuple(bincounts)

#%% private testparticle definitions
    
def _rdf_insertion_binned_3d_cuboid(coordinates,pairpotential,rmin=0,rmax=10,
    dr=None,boundary=None,pairpotential_binedges=None,n_ins=1000,
    interpolate=True,periodic_boundary=False,ins_coords=None,insert_grid=False,
    avoid_boundary=False,avoid_coordinates=False,neighbors_upper_bound=None,
    workers=1,combinations=None):
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

    #init function that returns energy from list of pairwise distances
    #if function is given, use that
    potfun = []
    for p in pairpotential:
        if callable(p):
            potfun.append(p)
        elif type(pairpotential_binedges) is None:
            pairpotential_binedges = rvals
        pairpotential_bincenter = (pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2
        
        if interpolate:#linearly interpolate pair potential between points
            pot_fun = lambda dist: np.interp(dist,pairpotential_bincenter,pairpotential)
        else:#get pair potential from nearest bin (round r to bincenter)
            from scipy.interpolate import interp1d    
            pot_fun = interp1d(pairpotential_bincenter,pairpotential,
                               kind='nearest',bounds_error=False,
                               fill_value='extrapolate')
    
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
        
        #add to overall lists
        pair_correlation += prob_r / prob_tot
        counter += counts.sum(axis=0)
      
    #take unweighted average of all datasets
    pair_correlation /= nf
    pair_correlation[counter==0] = 0
    
    return rvals,pair_correlation,counter


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