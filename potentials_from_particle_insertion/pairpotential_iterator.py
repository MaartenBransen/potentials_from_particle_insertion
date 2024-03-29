"""
This file contains functions to iteratively solve for the pair potential

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

#external imports
import numpy as np
from scipy.optimize import curve_fit

#internal imports
from .distribution_functions import (
    rdf_insertion_binned_3d,
    rdf_insertion_binned_2d,
    _get_rvals
)

#defs
def run_iteration(coordinates,pair_correlation_func,initial_guess=None,rmin=0,
                  rmax=10,dr=None,convergence_tol=1e-5,max_iterations=100,
                  zero_clip=1e-20,regulate=False,**kwargs):
    """Run the algorithm to solve for the pairwise potential that most 
    accurately reproduces the radial distribution function using test-particle
    insertion, based on the ideas in ref. [1].
    
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
    
    #create values for bin edges and centres of r
    rvals = _get_rvals(rmin,rmax,dr)
    rcent = (rvals[1:]+rvals[:-1])/2
    
    #check inputs
    if len(pair_correlation_func) != len(rcent):
        raise ValueError('lenght pair_correlation_func does not match rmax and dr')
    
    if type(initial_guess)==type(None):
        initial_guess = np.zeros(len(rcent))
    
    #check dimensionality, select appropriate rdf_insertion routine
    if type(coordinates)==list:
        dims = np.shape(coordinates[0])[1]
    else:
        dims = np.shape(coordinates)[1]
    
    if dims == 2:
        rdf_insertion_binned = rdf_insertion_binned_2d
    elif dims == 3:
        rdf_insertion_binned = rdf_insertion_binned_3d
    else:
        raise ValueError('array(s) in `coordinates` must have 2 or 3 columns')
      
    #set up for 0th iteration seperately
    pairpotential = [initial_guess]
    pair_correlation_func[pair_correlation_func<zero_clip]=zero_clip
    _,newpaircorrelation,c = rdf_insertion_binned(
        coordinates,
        initial_guess,
        rmin=rmin,
        rmax=rmax,
        dr=dr,
        **kwargs
    )
    newpaircorrelation[newpaircorrelation<zero_clip] = zero_clip#avoid deviding by zero
    paircorrelation = [newpaircorrelation]
    chi_squared = [np.mean((newpaircorrelation - pair_correlation_func)**2)]
    print('\riteration 0, χ²={:4g}'.format(chi_squared[-1]))
    
    #start the main iterative loop
    i = 1
    counters = [c]
    while i < max_iterations:
        
        #calculate the new pairwise potential, with regularisation
        if regulate:
            newpotential = _regulated_updater(
                    np.exp(-pairpotential[-1]),
                    np.exp(-pairpotential[-1])*pair_correlation_func/newpaircorrelation,
                    i#max(10,i)
                )
        else:
             newpotential = np.exp(-pairpotential[-1])*pair_correlation_func/newpaircorrelation   

        newpotential[newpotential<zero_clip] = zero_clip
        newpotential = -np.log(newpotential)
        pairpotential.append(newpotential)
        
        #use it to calculate the new g(r)
        _,newpaircorrelation,c = rdf_insertion_binned(
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
    
    return chi_squared,pairpotential,paircorrelation,counters

def run_iterator_fitfunction(coordinates,pair_correlation_func,potential_func,
        initial_guess=None,fit_bounds=None,rmin=0,rmax=20,dr=None,
        convergence_tol=1e-5,max_iterations=100,max_func_evals=100,
        zero_clip=1e-20,**kwargs):
    """Run the algorithm to solve for the pairwise potential that most 
    accurately reproduces the radial distribution function using test-particle
    insertion, using a fit function of known form instead of a generalized
    binned potential based on scipy.optimize.curve_fit.

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
    potential_func : callable
        fit funtion to use, where the first argument is the interparticle 
        distance r, and any subsequent arguments are optimized for.
    initial_guess : list, optional
        values for the function parameters to use for the first iteration. 
        The default is None.
    fit_bounds : 2-tuple of list_like, optional
        Each element of the tuple must be either an array with the length equal
        to the number of parameters, or a scalar (in which case the bound is 
        taken to be the same for all parameters). Use np.inf with an 
        appropriate sign to disable bounds on all or some parameters.
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
    max_func_evals : int, optional
        Maximum number of times the TPI function is evaluated. The default is 
        100.
    zero_clip : float, optional
        values below the value of zero-clip are set to this value to avoid
        devision by zero errors. The default is `1e-20`.
    **kwargs : key=value
        Additional keyword arguments are passed on to `rdf_insertion_binned_2d`
        or `rdf_insertion_binned_3d`

    Returns
    -------
    χ² : list of float
        summed squared error in the pair correlation function for each 
        iteration
    pairpotential : list of callable
        potential functions giving the potential for each iteraction
    fit_params : list of list of float
        the values for the function parameters in each iteration
    paircorrelation : list of list of float
        the values for the pair correlation function from test-particle
        insertion for each iteration
    counts : list of list of int
        number of pair counts contributing to each bin in each iteration

    See also
    --------
    rdf_insertion_binned_2d : 2D routine for g(r) from test-particle insertion
    rdf_insertion_binned_3d : 3D routine for g(r) from test-particle insertion

    """
    
    #create values for bin edges and centres of r
    rvals = _get_rvals(rmin,rmax,dr)
    rcent = (rvals[1:]+rvals[:-1])/2
    
    #check inputs
    if len(pair_correlation_func) != len(rcent):
        raise ValueError('lenght pair_correlation_func does not match rmax and dr')
    
    if type(fit_bounds) == type(None):
        fit_bounds = (-np.inf,np.inf)
    
    #check dimensionality, select appropriate rdf_insertion routine
    if type(coordinates)==list:
        dims = np.shape(coordinates[0])[1]
    else:
        dims = np.shape(coordinates)[1]
    
    if dims == 2:
        rdf_insertion_binned = rdf_insertion_binned_2d
    elif dims == 3:
        rdf_insertion_binned = rdf_insertion_binned_3d
    else:
        raise ValueError('array(s) in `coordinates` must have 2 or 3 columns')
    
    #set input pair correlation to zero-clip to avoid devision by 0 errors
    pair_correlation_func[pair_correlation_func<zero_clip]=zero_clip
    
    #initialize lists to write results to upon fitfunction call (to get 
    #intermediate results from scipy.optimize.curve_fit)
    paircorrelations = []
    pairpotentials = []
    fit_params = []
    chi_squared = []
    counters = []
    
    #generate fitfuntion from potential function and insertion routine
    def fitfun(r,*fitargs):
        
        #check max iterations
        if len(paircorrelations)>=max_func_evals:
            raise RuntimeError('The maximum number of function evaluations is reached')
        #calculate pairpotential function and call insertion routine
        pairpotential = lambda dist: potential_func(dist,*fitargs)
        _,newpaircorrelation,c = rdf_insertion_binned(
            coordinates,
            pairpotential,
            rmin=rmin,
            rmax=rmax,
            dr=dr,
            insert_grid=True,
            **kwargs
        )
        
        #avoid devision by zero errors
        newpaircorrelation[newpaircorrelation<zero_clip]=zero_clip
        
        #calculate error
        error = np.mean((newpaircorrelation-pair_correlation_func)**2)
        
        #write ouput of each iteration to lists upon fitfunction call
        paircorrelations.append(newpaircorrelation)
        pairpotentials.append(pairpotential)
        fit_params.append([*fitargs])
        chi_squared.append(error)
        counters.append(c)
        
        #print update
        print('function evaluation {:}, χ²={:4g}'.format(len(paircorrelations)-1,chi_squared[-1]))
        
        return newpaircorrelation
        #return error
    
    #fit
    try:
        fit,cov = curve_fit(
            fitfun, 
            rcent, 
            pair_correlation_func,
            p0=initial_guess,
            sigma=1/np.sqrt(pair_correlation_func),
            bounds = fit_bounds,
            jac='2-point',
            method='trf',
            max_nfev=max_iterations,
            xtol=None,
            gtol=None,
            ftol=convergence_tol,
            verbose=2,
        )
    except RuntimeError:
        print('max_iterations reached')
    
    return chi_squared,pairpotentials,fit_params,paircorrelations,counters

#%% private helper functions
def _regulated_updater(old,new,iteration):
    """returns weighted average of old and new with gradually shifting weight
    as iteration number increases to get ever more subtle changes"""
    if iteration<2:
        return new
    else:
        return np.average([old,new],axis=0,weights=[1,1/iteration**0.5])