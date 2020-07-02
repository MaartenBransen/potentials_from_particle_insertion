"""
Set of functions for calculating the radial distribution function using test-
particle insertion with a known pair potential.

Maarten Bransen, 2020
m.bransen@uu.nl
"""

#external imports
import numpy as np
import numba as nb
from scipy.spatial import cKDTree

#internal imports
from .generate_coordinates import _random_coordinates_in_box,\
    _random_coordinates_at_distance,_random_coordinates_on_sphere
from .geometry import _sphere_shell_vol_fraction

#defs
def rdf_insertion_binned_3d(coordinates,pairpotential,rmax,dr,boundary,
                            pairpotential_binedges=None,n_ins=1000,
                            interpolate=True,rmin=0,periodic_boundary=False,
                            avoid_boundary=False,avoid_coordinates=True):
    """Calculate g(r) from insertion of test-particles into sets of existing
    coordinates, averaged over bins of width dr, and based on the pairwise 
    interaction potential u(r) (in units of kT).

    Parameters
    ----------
    coordinates : iterable of (numpy.array of shape n*3)
        list of arrays of n particle coordinates, where each array is an 
        independent set (e.g. a time step from a video). In each array, 
        fist dimension elements must be of form `[z,y,x]`. Particles are 
        assumed to be inside of the bounding box defined by `boundary`. 
    pairpotential : iterable
        list of values for the pairwise interaction potential. Must have length
        of `len(pairpotential_binedges)-1` and be in units of thermal energy kT
    rmax : float
        cut-off radius for the pairwise distance (right edge of last bin).
    dr : float
        bin width of the pairwise distance bins.
    boundary : array-like of form `((zmin,zmax),(ymin,ymax),(xmin,xmax))`
        positions of the walls that define the bounding box of the coordinates.
    pairpotential_binedges : iterable, optional
        bin edges corresponding to the values in `pairpotential. The default 
        is None, which uses the bins defined by `rmin`, `rmax` and `dr`.
    n_ins : int, optional
        the number of test-particles to insert into each item in `coordinates`.
        The default is 1000.
    interpolate : bool, optional
        whether to use linear interpolation for calculating the interaction of 
        two particles using the values in `pairpotential`. The default is True.
    rmin : float, optional
        lower cut-off for the pairwise distance. The default is 0.
    periodic_boundary : TYPE, optional
        DESCRIPTION. The default is False.
    avoid_boundary : bool, optional
        if True, all test-particles are inserted at least `rmax` away from any 
        of the surfaces defined in `boundary` to avoid effects of the finite
        volume of the bounding box. The default is False, which, if
        `periodic_boundary=False`, uses an analytical correction factor for 
        missing volume of test-particles near the boundaries.
    avoid_coordinates : bool, optional
        whether to insert test-particles at least `rmin` away from the center 
        of any of the 'real' coordinates in `coordinates`. The default is True.

    Returns
    -------
    pair_correlation : TYPE
        DESCRIPTION.
    counter : TYPE
        DESCRIPTION.

    """

    #check input
    boundary = np.array(boundary)
    if avoid_boundary and rmax >= min(boundary[:,1]-boundary[:,0])/2:
        raise ValueError('rmax cannot be more than half the smallest box dimension')
    
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
    
    if avoid_boundary:
        reduced_boundary = boundary.copy()
        reduced_boundary[:,0] += rmax
        reduced_boundary[:,1] -= rmax
    
    #initialize arrays to store values
    counter = np.empty((nt,nr))
    pair_correlation = np.empty((nt,nr))
    
    #loop over all timesteps / independent sets of coordiates
    for i,coords in enumerate(coordinates):
        
        #generate new test-particle coordinates for each set
        if avoid_coordinates and avoid_boundary:
            trialparticles = _random_coordinates_at_distance(reduced_boundary,coords,rmin,n=n_ins)
        elif avoid_coordinates:
            trialparticles = _random_coordinates_at_distance(boundary,coords,rmin,n=n_ins)
        elif avoid_boundary:
            trialparticles = _random_coordinates_in_box(reduced_boundary,n=n_ins)
        else:
            trialparticles = _random_coordinates_in_box(boundary,n=n_ins)
         
        #init KDTree for fast pairfinding
        if periodic_boundary:
            coords -= boundary[:,0]#shift box to origin
            tree = cKDTree(coords,boxsize=boundary[:,1]-boundary[:,0])
        else:
            tree = cKDTree(coords)
        
        #find all pairs with one particle from testparticles and one from coordinates
        distances,_ = tree.query(trialparticles,k=len(coords),distance_upper_bound=rmax)
        
        #cKDTree pads rows with np.inf to get correct length, work with masked
        #arrays to only work on finite values
        mask = np.isfinite(distances) & (distances>0)
        distances = np.ma.masked_array(distances,mask)
        
        #calculate total potential energy of insertion (psi) for each testparticle
        #(row) by summing each pairwise potential energy u(r)
        if periodic_boundary or avoid_boundary:
            exp_psi = np.exp(-np.sum(pot_fun(distances)[mask],axis=1))
        
        else:
            #calculate correction factor for each testparticle for each dist bin
            # to account for missing information around particles near boundary
            boundarycorr = _sphere_shell_vol_fraction(
                rvals,
                boundary-trialparticles[:,:,np.newaxis]#shift coordinate system with origin in particle
                )
            
            #sum pairwise energy per particle per distance bin, then correct
            # each bin for missing volume, then sum and convert to probability 
            # e^(-psi)
            distances = np.ma.masked_array(distances,mask)
            exp_psi = np.apply_along_axis(
                lambda row: np.histogram(
                    row.data[row.mask],
                    bins=rvals,
                    weights=pot_fun(row.data[row.mask])
                    )[0],
                1,
                distances
                )
            
            exp_psi = np.exp(-np.sum(exp_psi/boundarycorr,axis=1))
            
        #calculate average probability of all testparticles
        prob_tot = np.mean(exp_psi)
        
        #count how many pairs each particle contributes to each bin, apply
        # boundary correction for correctly weighted average
        prob_r = np.apply_along_axis(lambda row: np.histogram(row.data[row.mask],bins=rvals)[0],1,distances)
        if not (periodic_boundary or avoid_boundary):
            prob_r = prob_r/boundarycorr 
        counts = prob_r.sum(axis=0)
        
        #take average of test-particle probabilities in each bin weighted by number of pair counts
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




def _calc_squared_dist(coordinates,trialparticle,rmax):
    #naive implementation
    distances = []
    for coord in coordinates:
        d = np.sum((coordinates-trialparticle)**2)
        if d <= rmax**2:
            distances.append(d)
    return distances

@nb.njit()
def _calc_squared_dist_numba(coordinates,trialparticle,rmax):
    """"numba optimized loop over coordinate pairs, returns pairwise distances"""
    #naive implementation
    distances = []
    for i in range(len(coordinates)):
        d = 0
        for j in range(3):
            d += (coordinates[i,j]-trialparticle[j])**2
        if d <= rmax**2:
            distances.append(d)
    return distances

def rdf_insertion_exact_3d(coordinates,pairpotential,rmax,dr,boundary,
                                 pairpotential_binedges=None,
                                 gen_prob_reps=1000,shell_prob_reps=10,
                                 interpolate=True,use_numba=True):
    """
    calculate g(r) from particle insertion method using particle coordinates
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
    
    if type(pairpotential_binedges) == type(None):
        pairpotential_binedges = rvals
    
    if interpolate:#interpolate pair potential between points
        pot_fun = lambda dist: np.interp(dist,(pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2,pairpotential)
    else:#get pair potential from nearest bin (round r to bincenter)
        pot_fun = lambda dist: np.array(pairpotential)[[np.digitize(dist,pairpotential_binedges)-1]]
    
    #calculate average probability over the whole box to insert particle
    tot_prob = 0
    for coords in coordinates:
        trialparticles = _random_coordinates_in_box(boundary,n=gen_prob_reps)
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
                trialcoords = _random_coordinates_on_sphere(
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

def pair_correlation_boundary_3d(coordinates,pairpotential,rmax,dr,boundary,
                             pairpotential_binedges=None,n_ins=1000,grid=False,
                             interpolate=True,rmin=0,periodic_boundary=False):
    """
    calculate g(r) from particle insertion method using particle coordinates
    and pairwise interaction potential u(r) (in units of kT)


    """
    global exp_psi,counts,distances
    
    #check input
    boundary = np.array(boundary)
    if rmax >= min(boundary[:,1]-boundary[:,0])/2:
        raise ValueError('rmax cannot be more than half the smallest box dimension')
    
    rvals = np.arange(rmin,rmax+dr,dr)#bins for r / u(r)
    rcent = rvals[:-1]+dr/2
    nt = len(coordinates)
    nr = len(rcent)
    
    if type(pairpotential_binedges) == type(None):
        pairpotential_binedges = rvals
    
    pairpotential_bincenter = (pairpotential_binedges[1:]+pairpotential_binedges[:-1])/2
    
    if interpolate:#interpolate pair potential between points
        pot_fun = lambda dist: np.interp(dist,pairpotential_bincenter,pairpotential)
    else:#get pair potential from nearest bin (round r to bincenter)
        from scipy.interpolate import interp1d    
        pot_fun = interp1d(pairpotential_bincenter,pairpotential,kind='nearest',bounds_error=False,fill_value='extrapolate')
    
    #set up reduced boundary for nonperiodic or box shifted to [0,boxize] for
    #periodic boundary conditions using scipy.cKDTree with boxsize argument
    reduced_boundary = boundary.copy()
    if periodic_boundary:
        reduced_boundary[:,0] -= boundary[:,0]
        reduced_boundary[:,1] -= boundary[:,0]
    else:
        reduced_boundary[:,0] += rmax
        reduced_boundary[:,1] -= rmax
    
    #generate n_ins coordinates in box on an organized grid
    if grid:
        numsteps = n_ins**(1/3)*(reduced_boundary[:,1]-reduced_boundary[:,0])/np.mean(reduced_boundary[:,1]-reduced_boundary[:,0])
        slices = [slice(dim[0],dim[1],1j*max([3,int(np.ceil(num))])) for dim,num in zip(reduced_boundary,numsteps)]
        trialparticles = np.mgrid[slices].reshape(3,-1).T
    
    #make arrays to store values
    counter = np.empty((nt,nr))
    pair_correlation = np.empty((nt,nr))
    
    #loop over all timesteps / independent sets of coordiates
    for i,coords in enumerate(coordinates):
        
        #generate new trialparticle coordinates
        if not grid:
            trialparticles = _random_coordinates_in_box(reduced_boundary,n=n_ins)
        
        #init KDTree for fast pairfinding
        if periodic_boundary:
            coords -= reduced_boundary[:,0]#shift box to origin
            tree = cKDTree(coords,boxsize=reduced_boundary[:,1])
        else:
            tree = cKDTree(coords)
        
        #find all pairs with one particle from testparticles and one from coordinates
        distances,_ = tree.query(trialparticles,k=len(coords),distance_upper_bound=rmax)

        #calculate sum of pairwise forces for each trialparticle
        mask = ~np.isfinite(distances)#cKDTree pads rows with np.inf to get correct length
        exp_psi = np.exp(-np.sum(np.ma.masked_array(pot_fun(distances),mask),axis=1))
        distances = np.ma.masked_array(distances,mask)
        
        #calculate probabilities
        prob_tot = np.mean(exp_psi)
        prob_r = np.apply_along_axis(lambda row: np.histogram(row.data[~row.mask],bins=rvals)[0],1,distances)
        counts = prob_r.sum(axis=0)
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

#@nb.njit()
