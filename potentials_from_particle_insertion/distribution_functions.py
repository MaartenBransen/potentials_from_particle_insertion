"""
Set of functions for calculating the radial distribution function using test-
particle insertion with a known pair potential.

Maarten Bransen, 2020
m.bransen@uu.nl
"""

#external imports
import numpy as np
from scipy.spatial import cKDTree

#check numba available
try:
    import numba as nb
    _numba_available = True
except ImportError:
    _numba_available = False

#internal imports
from .generate_coordinates import _rand_coord_in_box,_rand_coord_in_circle,\
    _rand_coord_at_dist,_rand_coord_on_sphere,_coord_grid_in_box
from .geometry import _sphere_shell_vol_fraction,_sphere_shell_vol_frac_periodic,\
    _circle_ring_area_fraction,_circle_ring_area_frac_periodic,\
    _circle_ring_area_fraction_circularboundary

#defs
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

def rdf_insertion_binned_3d(coordinates,pairpotential,rmax,dr,boundary,
                            pairpotential_binedges=None,n_ins=1000,
                            interpolate=True,rmin=0,periodic_boundary=False,
                            ins_coords=None,insert_grid=False,
                            avoid_boundary=False,avoid_coordinates=False,
                            neighbors_upper_bound=None):
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

    #assure array of arrays with first axis as dtype=object and rest floats
    if type(coordinates)==list:#weird syntax but there's no prettier way
        coordinates = np.array([None]+coordinates,dtype=object)[1:]
    
    elif type(coordinates)==np.ndarray:
        if not np.can_cast(coordinates[0].dtype,float):
            raise TypeError(
                "dtype `{}` of `coordinates` can't be broadcasted to `float`".format(coordinates[0].dtype)
            )
    else:
        raise TypeError(
            "dtype `{}` of `coordinates` not supported, use a list of numpy.array".format(type(coordinates))
        )
    
    #assure 3D array for coordinates
    if coordinates.ndim == 2:
        coordinates = coordinates[np.newaxis,:,:]

    #check if single boundary or boundary per dataset is given
    boundary = np.array(boundary)
    if boundary.ndim == 2:
        boundary = np.broadcast_to(boundary,(len(coordinates),3,2))
        
    #check if rmax input and boundary are feasible for avoiding boundary
    for bound in boundary:
        if avoid_boundary and rmax >= min(bound[:,1]-bound[:,0])/2:
            raise ValueError(
                'rmax cannot be more than half the smallest box dimension when '+
                'avoid_boundary=True, use rmax < {:}'.format(min(bound[:,1]-bound[:,0])/2)
                )
    
        #check rmax and boundary for edge-handling in periodic boundary conditions
        elif periodic_boundary:
            if min(bound[:,1]-bound[:,0])==max(bound[:,1]-bound[:,0]):
                boxlen = bound[0,1]-bound[0,0]
                if rmax > boxlen*np.sqrt(3)/2:
                    raise ValueError(
                        'rmax cannot be more than sqrt(3)/2 times the size of a '+
                        'cubic bounding box when periodic_boundary=True, use '+
                        'rmax < {:}'.format((bound[0,1]-bound[0,0])*np.sqrt(3)/2)
                    )
            elif rmax > min(bound[:,1]-bound[:,0]):
                raise NotImplementedError(
                    'rmax larger than half the smallest box dimension when '+
                    'periodic_boundary=True is only implemented for cubic boundaries'
                )
    
        #check rmax and boundary for edge handling without periodic boundaries
        else:
            if rmax > max(bound[:,1]-bound[:,0])/2:
                raise ValueError(
                    'rmax cannot be larger than half the largest dimension in '+
                    'boundary, use rmax < {:}'.format(max(bound[:,1]-bound[:,0])/2)
                )
    
    #create bin edges and bin centres for r
    rvals = np.arange(rmin,rmax+dr,dr)
    rcent = rvals[:-1]+dr/2
    nt = len(coordinates)
    nr = len(rcent)
    
    #init function that returns energy from list of pairwise distances
    #if function is given, use that
    if callable(pairpotential):
        pot_fun = pairpotential
    #otherwise use nearest or linearly interpolate
    else:
        #bin edges and centres for pairpotential
        if type(pairpotential_binedges) == type(None):
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
    counter = np.empty((nt,nr))
    pair_correlation = np.empty((nt,nr))
    
    #loop over all timesteps / independent sets of coordiates
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
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
        distances,_ = tree.query(trialparticles,k=k,distance_upper_bound=rmax)
        
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
                boundarycorr = _sphere_shell_vol_frac_periodic(
                    rvals,
                    min(bound[:,1]-bound[:,0])
                )[np.newaxis,:]
                
            else:
                #calculate correction factor for each testparticle for each dist bin
                # to account for missing information around particles near boundary.
                #boundary is shifted for coordinate system with origin in particle
                boundarycorr = _sphere_shell_vol_fraction(
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
            
        #calculate average probability of all testparticles
        prob_tot = np.mean(exp_psi)
        
        #count how many pairs each particle contributes to each bin
        if _numba_available:
            prob_r = _apply_hist_nb(distances,mask,rvals)
        else:
            prob_r = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                prob_r[j] = np.histogram(row[msk],bins=rvals)[0]
        
        #boundarycorrect probability counts for correct weighing
        if not avoid_boundary:
            prob_r = prob_r / boundarycorr 
        
        counts = prob_r.sum(axis=0)
        
        #take average of test-particle probabilities in each bin weighted by number
        # of pair counts
        prob_r = np.sum(prob_r * exp_psi[:,np.newaxis], axis=0)
        prob_r[counts!=0] /= counts[counts!=0]
        
        #store to lists
        counter[i] = counts
        pair_correlation[i] = prob_r/prob_tot
    
    pair_correlation = np.sum(pair_correlation*counter,axis=0)
    counter = counter.sum(axis=0)
    pair_correlation[counter!=0] /= counter[counter!=0]
    pair_correlation[counter==0] = 0
    
    return pair_correlation,counter


def rdf_insertion_binned_2d(coordinates,pairpotential,rmax,dr,boundary,
                            pairpotential_binedges=None,n_ins=1000,
                            interpolate=True,rmin=0,periodic_boundary=False,
                            avoid_boundary=False,avoid_coordinates=False,
                            neighbors_upper_bound=None):
    """Calculate g(r) from insertion of test-particles into sets of existing
    2D coordinates, averaged over bins of width dr, and based on the pairwise 
    interaction potential u(r) (in units of kT).
    
    Implementation partly based on ref. [1] but with novel corrections for 
    edge effects based on analytical formulas for periodic and nonperiodic 
    boundary conditions.

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
    
    """

    #assure array of arrays with first axis as dtype=object and rest floats
    if type(coordinates)==list:#weird syntax but there's no prettier way
        coordinates = np.array([None]+coordinates,dtype=object)[1:]
    
    elif type(coordinates)==np.ndarray:
        if not np.can_cast(coordinates[0].dtype,float):
            raise TypeError(
                "dtype `{}` of `coordinates` can't be broadcasted to `float`".format(coordinates[0].dtype)
            )
    else:
        raise TypeError(
            "dtype `{}` of `coordinates` not supported, use a list of numpy.array".format(type(coordinates))
        )
    
    #assure 3D array for coordinates
    if coordinates.ndim == 2:
        coordinates = coordinates[np.newaxis,:,:]

    #check if single boundary or boundary per coordinate set
    boundary = np.array(boundary)
    if boundary.ndim == 2:
        boundary = np.broadcast_to(boundary,(len(coordinates),2,2))
    
    #check if rmax input and boundary are feasible for avoiding boundary
    for bound in boundary:
        if avoid_boundary and rmax >= min(bound[:,1]-bound[:,0])/2:
            raise ValueError(
                'rmax cannot be more than half the smallest box dimension when '+
                'avoid_boundary=True, use rmax < {:}'.format(min(bound[:,1]-bound[:,0])/2)
            )
        
        #check rmax and boundary for edge-handling in periodic boundary conditions
        elif periodic_boundary:
            if bound[0,1]-bound[0,0] == bound[1,1]-bound[1,0]:
                boxlen = bound[0,1]-bound[0,0]
                if rmax > boxlen*np.sqrt(2)/2:
                    raise ValueError(
                        'rmax cannot be more than sqrt(2)/2 times the size of a '+
                        'square bounding box when periodic_boundary=True, use '+
                        'rmax < {:}'.format((bound[0,1]-bound[0,0])*np.sqrt(2)/2)
                    )
            elif rmax > min(bound[:,1]-bound[:,0]):
                raise NotImplementedError(
                    'rmax larger than half the smallest box dimension when '+
                    'periodic_boundary=True is only implemented for square boundaries'
                )
        
        #check rmax and boundary for edge handling without periodic boundaries
        else:
            if rmax > max(bound[:,1]-bound[:,0])/2:
                raise ValueError(
                    'rmax cannot be larger than half the largest dimension in '+
                    'boundary, use rmax < {:}'.format(max(bound[:,1]-bound[:,0])/2)
                )
    
    #create bin edges and bin centres for r
    rvals = np.arange(rmin,rmax+dr,dr)
    rcent = rvals[:-1]+dr/2
    nt = len(coordinates)
    nr = len(rcent)
    
    #bin edges and centres for pairpotential
    if type(pairpotential_binedges) == type(None):
        pairpotential_binedges = rvals
    pairpotential_bincenter = (pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2
    
    #init function that returns energy from list of pairwise distances
    if interpolate:#linearly interpolate pair potential between points
        pot_fun = lambda dist: np.interp(dist,pairpotential_bincenter,
                                         pairpotential)
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
    
    #initialize arrays to store values
    counter = np.empty((nt,nr))
    pair_correlation = np.empty((nt,nr))
    
    #loop over all timesteps / independent sets of coordiates
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #generate new test-particle coordinates for each set
        if avoid_coordinates and avoid_boundary:
            trialparticles = _rand_coord_at_dist(reduced_boundary[i],coords,rmin,n=n_ins)
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
        distances,_ = tree.query(trialparticles,k=k,distance_upper_bound=rmax)
        
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
            if periodic_boundary:
                #calculate correction factor for each distance bin to account 
                # for missing information
                boundarycorr = _circle_ring_area_frac_periodic(
                    rvals,
                    min(bound[:,1]-bound[:,0])
                )[np.newaxis,:]
                
            else:
                #calculate correction factor for each testparticle for each dist bin
                # to account for missing information around particles near boundary.
                #boundary is shifted for coordinate system with origin in particle
                boundarycorr = _circle_ring_area_fraction(
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
            
        #calculate average probability of all testparticles
        prob_tot = np.mean(exp_psi)
        
        #count how many pairs each particle contributes to each bin
        if _numba_available:
            prob_r = _apply_hist_nb(distances,mask,rvals)
        else:
            prob_r = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                prob_r[j] = np.histogram(row[msk],bins=rvals)[0]
        
        #boundarycorrect probability counts for correct weighing
        if not avoid_boundary:
            prob_r = prob_r / boundarycorr 
        
        counts = prob_r.sum(axis=0)
        
        #take average of test-particle probabilities in each bin weighted by number
        # of pair counts
        prob_r = np.sum(prob_r * exp_psi[:,np.newaxis], axis=0)
        prob_r[counts!=0] /= counts[counts!=0]
        
        #store to lists
        counter[i] = counts
        pair_correlation[i] = prob_r/prob_tot
    
    pair_correlation = np.sum(pair_correlation*counter,axis=0)
    counter = counter.sum(axis=0)
    pair_correlation[counter!=0] /= counter[counter!=0]
    pair_correlation[counter==0] = 0
    
    return pair_correlation,counter

def rdf_insertion_exact_3d(coordinates,pairpotential,rmax,dr,boundary,
                                 pairpotential_binedges=None,
                                 gen_prob_reps=1000,shell_prob_reps=10,
                                 interpolate=True,use_numba=True):
    """calculate g(r) from particle insertion method using particle coordinates
    and pairwise interaction potential u(r) (in units of kT). Inserts test-
    particles at a specific r for every real particle

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

    """
    rvals = np.arange(0,rmax+dr,dr)#bins for r / u(r)
    rcent = rvals[:-1]+dr/2
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

def _calc_squared_dist(coordinates,trialparticle,rmax):
    """pure python loop over coordinate pairs, returns pairwise distances"""
    #naive implementation, slow
    distances = []
    for coord in coordinates:
        d = np.sum((coordinates-trialparticle)**2)
        if d <= rmax**2:
            distances.append(d)
    return distances

if _numba_available:
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


def rdf_dist_hist_3d(coordinates,rmin=0,rmax=10,dr=None,boundary=None,
                     density=None,periodic_boundary=False,handle_edge=True,
                     quiet=False,neighbors_upper_bound=None):
    """calculates g(r) via a 'conventional' distance histogram method for a 
    set of 3D coordinate sets. Provided for convenience. Edge correction based
    on refs [1] and [2].

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
    rvals = np.arange(rmin,rmax+dr,dr)
    
    #assure array of arrays with first axis as dtype=object and rest floats
    if type(coordinates)==list:#weird syntax but there's no prettier way
        coordinates = np.array([None]+coordinates,dtype=object)[1:]
    
    elif type(coordinates)==np.ndarray:
        if not np.can_cast(coordinates[0].dtype,float):
            raise TypeError(
                "dtype `{}` of `coordinates` can't be broadcasted to `float`".format(coordinates[0].dtype)
            )
    else:
        raise TypeError(
            "dtype `{}` of `coordinates` not supported, use a list of numpy.array".format(type(coordinates))
        )
    
    #assure 3D array for coordinates
    if coordinates.ndim == 2:
        coordinates = coordinates[np.newaxis,:,:]
    
    #set default step size
    if type(dr)==type(dr):
        dr = (rmax-rmin)/20
    
    #set default boundary as min and max values in dataset
    if type(boundary)==type(None):
        boundary = np.array([
                [coordinates[:,:,0].min(),coordinates[:,:,0].max()],
                [coordinates[:,:,1].min(),coordinates[:,:,1].max()],
                [coordinates[:,:,2].min(),coordinates[:,:,2].max()]
            ]*len(coordinates))
    else:
        boundary = np.array(boundary)
        if boundary.ndim == 2:
            boundary = np.broadcast_to(boundary, (len(coordinates),3,2))
    
    #check rmax and boundary for edge-handling in periodic boundary conditions
    if periodic_boundary:
        for bound in boundary:
            if min(bound[:,1]-bound[:,0])==max(bound[:,1]-bound[:,0]):
                boxlen = bound[0,1]-bound[0,0]
                if rmax > boxlen*np.sqrt(3)/2:
                    raise ValueError(
                        'rmax cannot be more than sqrt(3)/2 times the size of a '+
                        'cubic bounding box when periodic_boundary=True, use '+
                        'rmax < {:}'.format((bound[0,1]-bound[0,0])*np.sqrt(3)/2)
                    )
            elif rmax > min(bound[:,1]-bound[:,0]):
                raise NotImplementedError(
                    'rmax larger than half the smallest box dimension when '+
                    'periodic_boundary=True is only implemented for cubic boundaries'
                )
    
    #check rmax and boundary for edge handling without periodic boundaries
    else:
        for bound in boundary:
            if rmax > max(bound[:,1]-bound[:,0])/2:
                raise ValueError(
                    'rmax cannot be larger than half the largest dimension in '+
                    'boundary, use rmax < {:}'.format(max(bound[:,1]-bound[:,0])/2)
                )
    
    #set density to mean number density in dataset
    if not density:
        vol = np.product(boundary[:,:,1]-boundary[:,:,0],axis=1)
        density = np.mean([len(coords)/v for v,coords in zip(vol,coordinates)])
    
    #loop over all sets of coordinates
    bincounts = []
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #print progress
        if not quiet:
            print('\rcalculating distance histogram g(r) {:} of {:}'.format(i+1,len(coordinates)),end='')
        
        if periodic_boundary:
            #set up KDTree for fast neighbour finding
            #shift box boundary corner to origin for periodic KDTree
            tree = cKDTree(coords-bound[:,0],boxsize=bound[:,1]-bound[:,0])
            
            #count number of particle pairs per bin directly
            counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
            
            #optionally apply edge correction to each bin
            if handle_edge:
                boundarycorr = _sphere_shell_vol_frac_periodic(
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
                dist,indices = tree.query(coords,k=k,distance_upper_bound=rmax)
            
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

                boundarycorr=_sphere_shell_vol_fraction(
                    rvals,
                    bound-coords[:,:,np.newaxis]
                    )
                counts = np.sum(counts/boundarycorr,axis=0)
        
            #otherwise calculate neighbours per bin directly
            else:
                counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
        
        #normalize and add to overall list
        #counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
        bincounts.append(counts / (4/3*np.pi * (rvals[1:]**3 - rvals[:-1]**3)) / (density*len(coords)))
        
    #newline
    if not quiet:
        print()
    
    #average all datasets
    bincounts = np.mean(bincounts,axis=0)
    
    return rvals,bincounts


def rdf_dist_hist_2d(coordinates,rmin=0,rmax=10,dr=None,boundary=None,
                     density=None,periodic_boundary=False,handle_edge=True,
                     quiet=False,neighbors_upper_bound=None):
    """calculates g(r) via a 'conventional' distance histogram method for a 
    set of 2D coordinate sets. Provided for convenience. Edge correction is 
    fully analytical for both periodic and nonperiodic boundary conditions.

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

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    bincounts : numpy.array
        values for the bins of the radial distribution function
        
    """
    #create bins
    rvals = np.arange(rmin,rmax+dr,dr)
    
    #assure array of arrays with first axis as dtype=object and rest floats
    if type(coordinates)==list:#weird syntax but there's no prettier way
        coordinates = np.array([None]+coordinates,dtype=object)[1:]
    elif type(coordinates)==np.ndarray:
        if not np.can_cast(coordinates[0].dtype,float):
            raise TypeError(
                "dtype `{}` of `coordinates` can't be broadcasted to `float`".format(coordinates[0].dtype)
            )
    else:
        raise TypeError(
            "dtype `{}` of `coordinates` not supported, use a list of numpy.array".format(type(coordinates))
        )
        
    #set default step size
    if type(dr)==type(dr):
        dr = (rmax-rmin)/20
        
    #check list of coordinates or only one coordinate set
    if coordinates.ndim == 2:
        coordinates = coordinates[np.newaxis,:,:]
    
    #set default boundary as min and max values in dataset
    if type(boundary)==type(None):
        boundary = np.array([
                [coordinates[:,:,0].min(),coordinates[:,:,0].max()],
                [coordinates[:,:,1].min(),coordinates[:,:,1].max()]
            ]*len(coordinates))
    else:
        boundary = np.array(boundary)
        
        #assure list of coordinate sets
        if boundary.ndim==2:
            boundary = np.array([boundary]*len(coordinates))
    
    
    #check rmax and boundary for edge-handling in periodic boundary conditions
    if periodic_boundary:
        for bound in boundary:
            if bound[0,1]-bound[0,0] == bound[1,1]-bound[1,0]:
                boxlen = bound[0,1]-bound[0,0]
                if rmax > boxlen*np.sqrt(2)/2:
                    raise ValueError(
                        'rmax cannot be more than sqrt(2)/2 times the size of a '+
                        'square bounding box when periodic_boundary=True, use '+
                        'rmax < {:}'.format((bound[0,1]-bound[0,0])*np.sqrt(2)/2)
                    )
            elif rmax > min(bound[:,1]-bound[:,0])/2:
                raise NotImplementedError(
                    'rmax larger than half the smallest box dimension when '+
                    'periodic_boundary=True is only implemented for square boundaries'
                )
    
    #check rmax and boundary for edge handling without periodic boundaries
    else:
        for bound in boundary:
            if rmax > max(bound[:,1]-bound[:,0])/2:
                raise ValueError(
                    'rmax cannot be larger than half the largest dimension in '+
                    'boundary, use rmax < {:}'.format(max(bound[:,1]-bound[:,0])/2)
                )
    
    #set density to mean number density in dataset
    if not density:
        vol = np.product(boundary[:,:,1]-boundary[:,:,0],axis=1)
        density = np.mean([len(coords)/v for v,coords in zip(vol,coordinates)])
    
    
    #loop over all sets of coordinates
    bincounts = []
    for i,(bound,coords) in enumerate(zip(boundary,coordinates)):
        
        #print progress
        if not quiet:
            print('\rcalculating distance histogram g(r) {:d} of {:d}'.format(i+1,len(coordinates)),end='')
        
        #in case of periodic boundary conditions
        if periodic_boundary:
            #set up KDTree for fast neighbour finding
            #shift box boundary corner to origin for periodic KDTree
            tree = cKDTree(coords-bound[:,0],boxsize=bound[:,1]-bound[:,0])
            
            #count number of particle pairs per bin directly
            counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
            
            #optionally apply edge correction for distances beyond boxsize/2
            if handle_edge:
                
                boundarycorr = _circle_ring_area_frac_periodic(
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
                dist,indices = tree.query(coords,k=k,distance_upper_bound=rmax)
                
                #remove pairs with self, padded (infinite) values and anythin below rmin
                dist = dist[:,1:]
                mask = np.isfinite(dist) & (dist>=rmin)
                
                #histogram the distances per reference particle and apply correction factor
                #for missing volume to each particle (each row) and each distance bin separately
                if _numba_available:
                    counts = _apply_hist_nb(dist,mask,rvals)
                else:
                    counts = np.zeros((len(dist),len(rvals)-1))
                    for j,(row,msk) in enumerate(zip(dist,mask)):
                        counts[j] = np.histogram(row[msk],bins=rvals)[0]
                boundarycorr=_circle_ring_area_fraction(
                    rvals,
                    bound-coords[:,:,np.newaxis]
                    )
                counts = np.sum(counts/boundarycorr,axis=0)
                
            #otherwise find and bin pairs directly
            else:
                counts = tree.count_neighbors(tree,rvals,cumulative=False)[1:]
        
        #normalize and add to overall list
        bincounts.append(counts / (np.pi*(rvals[1:]**2 - rvals[:-1]**2)) / (density*len(coords)))
    
    #newline
    if not quiet:
        print()
    
    #average all datasets
    bincounts = np.mean(bincounts,axis=0)
    
    return rvals,bincounts

def rdf_dist_hist_2d_circularboundary(coordinates,boundary_pos,boundary_rad,
                                      rmin=0,rmax=10,dr=None,density=None,
                                      quiet=False,neighbors_upper_bound=None):
    """calculates g(r) via a 'conventional' distance histogram method for a 
    set of 2D coordinate sets. Provided for convenience. Edge correction is 
    fully analytical for both periodic and nonperiodic boundary conditions.

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
    boundary_pos : tuple of floats
        coordinates of the centre of the circle defining the boundary as 
        `(y,x)` tuple or array-like, or list like of such.
    boundary_rad : float
        radius defining the circular boundary, or list of floats for each 
        coordinate set separately
    rmin : float, optional
        lower bound for the pairwise distance, left edge of 0th bin. The 
        default is 0.
    rmax : float, optional
        upper bound for the pairwise distance, right edge of last bin. The 
        default is 10.
    dr : float, optional
        bin width for the pairwise distance bins. The default is (rmax-rmin)/20
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

    Returns
    -------
    rvals : numpy.array
        bin-edges of the radial distribution function.
    bincounts : numpy.array
        values for the bins of the radial distribution function
        
    """
    #create bins
    rvals = np.arange(rmin,rmax+dr,dr)
    
    #assure array of arrays with first axis as dtype=object and rest floats
    if type(coordinates)==list:#weird syntax but there's no prettier way
        coordinates = np.array([None]+coordinates,dtype=object)[1:]
    elif type(coordinates)==np.ndarray:
        if not np.can_cast(coordinates[0].dtype,float):
            raise TypeError(
                "dtype `{}` of `coordinates` can't be broadcasted to `float`".format(coordinates[0].dtype)
            )
    else:
        raise TypeError(
            "dtype `{}` of `coordinates` not supported, use a list of numpy.array".format(type(coordinates))
        )
    
    #check list of coordinates or only one coordinate set
    if coordinates.ndim == 2:
        coordinates = coordinates[np.newaxis,:,:]
    
    #same for boundaries
    if type(boundary_rad) != list:
        boundary_rad = [boundary_rad]*len(coordinates)
    if type(boundary_pos[0]) not in (list,np.ndarray,tuple):
        boundary_pos = [boundary_pos]*len(coordinates)
    
    #set default step size
    if type(dr)==type(dr):
        dr = (rmax-rmin)/20
    
    #check rmax and boundary for edge handling
    if any(rmax >= 2*np.array(boundary_rad)):
        raise ValueError('rmax cannot be larger than 2*boundary_rad')
    
    #set density to mean number density in dataset
    if not density:
        vol = np.pi*np.array(boundary_rad)**2
        density = np.mean([len(coords)/v for v,coords in zip(vol,coordinates)])
    
    
    #loop over all sets of coordinates
    bincounts = []
    for i,(bpos,brad,coords) in enumerate(zip(boundary_pos,boundary_rad,coordinates)):
        
        #print progress
        if not quiet:
            print('\rcalculating distance histogram g(r) {:d} of {:d}'.format(i+1,len(coordinates)),end='')
         
        #set default neighbor_upper_bound
        if type(neighbors_upper_bound)==type(None):
            k = len(coords)
        else:
            k = min([neighbors_upper_bound,len(coords)])
         
        #set up and query tree for fast neighbor finding
        tree = cKDTree(coords)
        dist,indices = tree.query(coords,k=k,distance_upper_bound=rmax)
        
        #remove pairs with self, padded (infinite) values and anythin below rmin
        dist = dist[:,1:]
        mask = np.isfinite(dist) & (dist>=rmin)
        
        #histogram the distances per reference particle and apply correction factor
        #for missing volume to each particle (each row) and each distance bin separately
        if _numba_available:
            counts = _apply_hist_nb(dist,mask,rvals)
        else:
            counts = np.zeros((len(dist),len(rvals)-1))
            for j,(row,msk) in enumerate(zip(dist,mask)):
                counts[j] = np.histogram(row[msk],bins=rvals)[0]
        boundarycorr=_circle_ring_area_fraction_circularboundary(
            rvals,
            np.sqrt(np.sum((coords - np.array(bpos))**2,axis=1)),
            brad
        )
        counts = np.sum(counts/boundarycorr,axis=0)
        
        #normalize and add to overall list
        bincounts.append(counts / (np.pi*(rvals[1:]**2 - rvals[:-1]**2)) / (density*len(coords)))
    
    #newline
    if not quiet:
        print()
    
    #average all datasets
    bincounts = np.mean(bincounts,axis=0)
    
    return rvals,bincounts

def rdf_insertion_binned_2d_circularboundary(coordinates,pairpotential,rmax,dr,
                                             boundary_pos,boundary_rad,
                                             pairpotential_binedges=None,
                                             n_ins=1000,interpolate=True,
                                             rmin=0,periodic_boundary=False,
                                             avoid_boundary=False,
                                             neighbors_upper_bound=None):
    """Calculate g(r) from insertion of test-particles into sets of existing
    2D coordinates, averaged over bins of width dr, and based on the pairwise 
    interaction potential u(r) (in units of kT).
    
    Implementation partly based on ref. [1] but with novel corrections for 
    edge effects based on analytical formulas for periodic and nonperiodic 
    boundary conditions.

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
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.

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
    
    """

    #assure array of arrays with first axis as dtype=object and rest floats
    if type(coordinates)==list:#weird syntax but there's no prettier way
        coordinates = np.array([None]+coordinates,dtype=object)[1:]
    
    elif type(coordinates)==np.ndarray:
        if not np.can_cast(coordinates[0].dtype,float):
            raise TypeError(
                "dtype `{}` of `coordinates` can't be broadcasted to `float`".format(coordinates[0].dtype)
            )
    else:
        raise TypeError(
            "dtype `{}` of `coordinates` not supported, use a list of numpy.array".format(type(coordinates))
        )
    
    #assure 3D array for coordinates
    if coordinates.ndim == 2:
        coordinates = coordinates[np.newaxis,:,:]

    #check if single boundary or boundary per coordinate set
    if type(boundary_rad) not in (list,np.ndarray,tuple):
        boundary_rad = [boundary_rad]*len(coordinates)
        boundary_pos = [boundary_pos]*len(coordinates)

    
    #check if rmax input and boundary are feasible for avoiding boundary
    if rmax > min(boundary_rad):
        raise NotImplementedError(
            'rmax cannot be larger the smallest value in boundary_rad'
        )
    
    #create bin edges and bin centres for r
    rvals = np.arange(rmin,rmax+dr,dr)
    rcent = rvals[:-1]+dr/2
    nt = len(coordinates)
    nr = len(rcent)
    
    #bin edges and centres for pairpotential
    if type(pairpotential_binedges) == type(None):
        pairpotential_binedges = rvals
    pairpotential_bincenter = (pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2
    
    #init function that returns energy from list of pairwise distances
    if interpolate:#linearly interpolate pair potential between points
        pot_fun = lambda dist: np.interp(dist,pairpotential_bincenter,
                                         pairpotential)
    else:#get pair potential from nearest bin (round r to bincenter)
        from scipy.interpolate import interp1d    
        pot_fun = interp1d(pairpotential_bincenter,pairpotential,
                           kind='nearest',bounds_error=False,
                           fill_value='extrapolate')
    
    #initialize arrays to store values
    counter = np.empty((nt,nr))
    pair_correlation = np.empty((nt,nr))
    
    #loop over all timesteps / independent sets of coordiates
    for i,(boundpos,boundrad,coords) in enumerate(zip(boundary_pos,boundary_rad,coordinates)):
        
        #generate new test-particle coordinates for each set
        if avoid_boundary:
            trialparticles = _rand_coord_in_circle(boundpos,boundrad-rmax,n=n_ins)
        else:
            trialparticles = _rand_coord_in_circle(boundpos,boundrad,n=n_ins)
         
        #init KDTree for fast pairfinding
        tree = cKDTree(coords)
        
        #find all pairs with one particle from testparticles and one from coordinates
        distances,_ = tree.query(trialparticles,k=len(coords),distance_upper_bound=rmax)
        
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
            boundarycorr = _circle_ring_area_fraction_circularboundary(
                rvals,
                np.sqrt(np.sum((trialparticles - np.array(boundpos))**2,axis=1)),
                boundrad
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
            
        #calculate average probability of all testparticles
        prob_tot = np.mean(exp_psi)
        
        #count how many pairs each particle contributes to each bin
        if _numba_available:
            prob_r = _apply_hist_nb(distances,mask,rvals)
        else:
            prob_r = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                prob_r[j] = np.histogram(row[msk],bins=rvals)[0]
        
        #boundarycorrect probability counts for correct weighing
        if not avoid_boundary:
            prob_r = prob_r / boundarycorr 
        
        counts = prob_r.sum(axis=0)
        
        #take average of test-particle probabilities in each bin weighted by number
        # of pair counts
        prob_r = np.sum(prob_r * exp_psi[:,np.newaxis], axis=0)
        prob_r[counts!=0] /= counts[counts!=0]
        
        #store to lists
        counter[i] = counts
        pair_correlation[i] = prob_r/prob_tot
    
    pair_correlation = np.sum(pair_correlation*counter,axis=0)
    counter = counter.sum(axis=0)
    pair_correlation[counter!=0] /= counter[counter!=0]
    pair_correlation[counter==0] = 0
    
    return pair_correlation,counter

def rdf_insertion_binned_2d_customboundary(coordinates,pairpotential,rmax,dr,
                            boundary,boundary_func,testparticle_func,
                            pairpotential_binedges=None,n_ins=1000,
                            interpolate=True,rmin=0,avoid_coordinates=False,
                            neighbors_upper_bound=None):
    """Calculate g(r) from insertion of test-particles into sets of existing
    2D coordinates, averaged over bins of width dr, and based on the pairwise 
    interaction potential u(r) (in units of kT).
    
    Implementation partly based on ref. [1] but with novel corrections for 
    edge effects based on analytical formulas for periodic and nonperiodic 
    boundary conditions.

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
    rmin : float, optional
        lower cut-off for the pairwise distance. The default is 0.
    avoid_coordinates : bool, optional
        whether to insert test-particles at least `rmin` away from the center 
        of any of the 'real' coordinates in `coordinates`. The default is 
        False.
    neighbors_upper_bound : int, optional
        upper limit on the number of neighbors expected within rmax from a 
        particle. Useful for datasets with dimensions much larger than rmax.
        Only relevant for the case where `handle_edge=True`. The default is 
        None, which takes the number of particles in the stack.

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
    
    """

    #assure array of arrays with first axis as dtype=object and rest floats
    if type(coordinates)==list:#weird syntax but there's no prettier way
        coordinates = np.array([None]+coordinates,dtype=object)[1:]
    
    elif type(coordinates)==np.ndarray:
        if not np.can_cast(coordinates[0].dtype,float):
            raise TypeError(
                "dtype `{}` of `coordinates` can't be broadcasted to `float`".format(coordinates[0].dtype)
            )
    else:
        raise TypeError(
            "dtype `{}` of `coordinates` not supported, use a list of numpy.array".format(type(coordinates))
        )
    
    if len(coordinates) != len(boundary):
        raise ValueError('lengths of boundary and coordinates must match')
    if len(coordinates) != len(boundary_func):
        raise ValueError('lengths of boundary_func and coordinates must match')
    if len(coordinates) != len(testparticle_func):
        raise ValueError('lengths of testparticle_func and coordinates must match')
    
    #assure 3D array for coordinates
    if coordinates.ndim == 2:
        coordinates = coordinates[np.newaxis,:,:]
    
    #create bin edges and bin centres for r
    rvals = np.arange(rmin,rmax+dr,dr)
    rcent = rvals[:-1]+dr/2
    nt = len(coordinates)
    nr = len(rcent)
    
    #bin edges and centres for pairpotential
    if type(pairpotential_binedges) == type(None):
        pairpotential_binedges = rvals
    pairpotential_bincenter = (pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2
    
    #init function that returns energy from list of pairwise distances
    if interpolate:#linearly interpolate pair potential between points
        pot_fun = lambda dist: np.interp(dist,pairpotential_bincenter,
                                         pairpotential)
    else:#get pair potential from nearest bin (round r to bincenter)
        from scipy.interpolate import interp1d    
        pot_fun = interp1d(pairpotential_bincenter,pairpotential,
                           kind='nearest',bounds_error=False,
                           fill_value='extrapolate')
    
    #initialize arrays to store values
    counter = np.empty((nt,nr))
    pair_correlation = np.empty((nt,nr))
    
    #loop over all timesteps / independent sets of coordiates
    for i,(bound,boundfunc,testfunc,coords) in \
        enumerate(zip(boundary,boundary_func,testparticle_func,coordinates)):
        
        #generate new test-particle coordinates for each set
        if avoid_coordinates:
            trialparticles = testfunc(bound,coords,rmin,n=n_ins)
        else:
            trialparticles = testfunc(bound,n=n_ins)
        
        #init KDTree for fast pairfinding
        tree = cKDTree(coords)
        
        #set default neighbor_upper_bound
        if type(neighbors_upper_bound)==type(None):
            k = len(coords)
        else:
            k = min([neighbors_upper_bound,len(coords)])
        
        #find all pairs with one particle from testparticles and one from coordinates
        distances,_ = tree.query(trialparticles,k=k,distance_upper_bound=rmax)
        
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
            
        #calculate average probability of all testparticles
        prob_tot = np.mean(exp_psi)
        
        #count how many pairs each particle contributes to each bin
        if _numba_available:
            prob_r = _apply_hist_nb(distances,mask,rvals)
        else:
            prob_r = np.empty((n_ins,nr))
            for j,(row,msk) in enumerate(zip(distances,mask)):
                prob_r[j] = np.histogram(row[msk],bins=rvals)[0]
        
        #boundarycorrect probability counts for correct weighing
        prob_r = prob_r / boundarycorr 
        
        counts = prob_r.sum(axis=0)
        
        #take average of test-particle probabilities in each bin weighted by number
        # of pair counts
        prob_r = np.sum(prob_r * exp_psi[:,np.newaxis], axis=0)
        prob_r[counts!=0] /= counts[counts!=0]
        
        #store to lists
        counter[i] = counts
        pair_correlation[i] = prob_r/prob_tot
    
    pair_correlation = np.sum(pair_correlation*counter,axis=0)
    counter = counter.sum(axis=0)
    pair_correlation[counter!=0] /= counter[counter!=0]
    pair_correlation[counter==0] = 0
    
    return pair_correlation,counter