"""
Set of functions for determining the missing volume of a spherical shell which
falls partially outside of a cuboidal boundary.

Maarten Bransen, 2020
m.bransen@uu.nl
"""

#external imports
import numpy as np

#internal imports
from .distribution_functions import rdf_insertion_binned_3d

#defs
def run_iteration(coordinates,pair_correlation_func,boundary,
                  initial_guess=None,rmin=0,rmax=20,dr=0.5,convergence_tol=0.1,
                  max_iterations=100,zero_clip=1e-10,regulate=False,**kwargs):
    """
    Run the algorithm to solve for the pairwise potential that most accurately
    reproduces the radial distribution function using test-particle insertion

    Parameters
    ----------
    coordinates : list-like of numpy.array
        List of sets of coordinates, where each item along the 0th dimension is
        a n*3 numpy.array of particle coordinates, where each array is an 
        independent set of coordinates (e.g. one z-stack, a time step from a 
        video, etc.), with each element of the array of form  `[z,y,x]`. Each 
        set of coordinates is not required to have the same number of particles
        but all stacks must share the same  bounding box as given by 
        `boundary`, and all coordinates must be within this bounding box.
    pair_correlation_func : list of float
        bin values for the true pair correlation function that the algorithm 
        will try to match iteratively.
    boundary : array-like of form `((zmin,zmax),(ymin,ymax),(xmin,xmax))`
        positions of the walls that define the bounding box of the coordinates.
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
        considered to be converged and ended. The default is 0.1.
    max_iterations : int, optional
        Maximum number of iterations after which the algorithm is ended. The
        default is 100.
    regulate : bool, optional
        if True, use regularization to more gently nudge towards the input g(r)
        at the cost of slower convergence. Experimental option. The default is
        False.
    **kwargs : key=value
        Additional keyword arguments are passed on to `rdf_insertion_binned_3d`

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

    See also
    --------
    `rdf_insertion_binned_3d`

    """
    
    #create values for bin edges and centres of r
    rvals = np.arange(rmin,rmax+dr,dr)
    rcent = rvals[:-1]+dr/2
    
    #check inputs
    if len(pair_correlation_func) != len(rcent):
        raise ValueError('lenght pair_correlation_func does not match rmax and dr')
    
    if type(initial_guess)==type(None):
        initial_guess = np.zeros(len(rcent))
    
    
    pair_correlation_func[pair_correlation_func<zero_clip]=zero_clip
    
    #set up for 0th iteration seperately
    pairpotential = [initial_guess]
    newpaircorrelation,_ = rdf_insertion_binned_3d(coordinates,initial_guess,
                                                   rmax,dr,boundary,rmin=rmin,
                                                   **kwargs)
    newpaircorrelation[newpaircorrelation<zero_clip] = zero_clip#avoid deviding by zero
    paircorrelation = [newpaircorrelation]
    chi_squared = [np.mean((newpaircorrelation - pair_correlation_func)**2)]
    print('\riteration 0, χ²={:4g}'.format(chi_squared[-1]))
    
    #start the main iterative loop
    i = 1
    counters = []
    while i < max_iterations:
        
        #calculate the new pairwise potential, with relaxation
        if regulate:
            newpotential = np.average(
                [
                    np.exp(-pairpotential[-1]),
                    np.exp(-pairpotential[-1])*pair_correlation_func/newpaircorrelation
                ],
                axis=0,
                weights=[i,max_iterations-i]
                )
        else:
             newpotential = np.exp(-pairpotential[-1])*pair_correlation_func/newpaircorrelation   

        newpotential[newpotential<zero_clip] = zero_clip
        newpotential = -np.log(newpotential)
        pairpotential.append(newpotential)
        
        #use it to calculate the new g(r)
        newpaircorrelation,c = rdf_insertion_binned_3d(
            coordinates,
            newpotential,
            rmax,
            dr,
            boundary,
            rmin=rmin,
            **kwargs)
        counters.append(c)
        newpaircorrelation[newpaircorrelation<1e-10] = 1e-10
        #newpaircorrelation[newpaircorrelation>20] = 20
        paircorrelation.append(newpaircorrelation)
        
        #calculate summed squared error
        chi_squared.append(np.mean((newpaircorrelation - pair_correlation_func)**2))
        
        print('\riteration {:}, χ²={:4g}'.format(i,chi_squared[-1]))
        if chi_squared[-1] < convergence_tol:
            break
        
        i += 1
    
    return chi_squared,pairpotential,paircorrelation,counters