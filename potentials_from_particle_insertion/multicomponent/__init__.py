#%%imports

import numpy as np
from warnings import warn
from scipy.spatial import cKDTree
from ..distribution_functions import \
    _numba_available,_get_rvals,\
    _apply_hist_nb
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

#%% private definitions

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
    if density is None:
        vol = np.product(boundary[:,:,1]-boundary[:,:,0],axis=1)
        density = [
            np.mean([len(coords[c])/v for v,coords in zip(vol,coordinates)]) \
                for c in range(nc)]
    
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