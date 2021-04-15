"""
Iterator to solve for pairwise potential from a set of coordinates

Maarten Bransen, 2020
m.bransen@uu.nl
"""

#external imports
import numpy as np
from scipy.optimize import curve_fit

#internal imports
from .distribution_functions import rdf_insertion_binned_3d,rdf_insertion_binned_2d,\
    rdf_insertion_binned_2d_circularboundary,rdf_insertion_binned_2d_customboundary

#defs
def run_iteration(coordinates,pair_correlation_func,boundary,
                  initial_guess=None,rmin=0,rmax=20,dr=0.5,
                  convergence_tol=1e-5,max_iterations=100,zero_clip=1e-20,
                  regulate=False,**kwargs):
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
    rvals = np.arange(rmin,rmax+dr,dr)
    rcent = rvals[:-1]+dr/2
    
    #check inputs
    if len(pair_correlation_func) != len(rcent):
        raise ValueError('lenght pair_correlation_func does not match rmax and dr')
    
    if type(initial_guess)==type(None):
        initial_guess = np.zeros(len(rcent))
    
    #check dimensionality, select appropriate rdf_insertion routine
    boundary = np.array(boundary)
    if boundary.shape[-2] == 2:
        rdf_insertion_binned = rdf_insertion_binned_2d
    elif boundary.shape[-2] == 3:
        rdf_insertion_binned = rdf_insertion_binned_3d
    else:
        raise ValueError('data and boundaries must be 2- or 3-dimensional')
      
    #set up for 0th iteration seperately
    pairpotential = [initial_guess]
    pair_correlation_func[pair_correlation_func<zero_clip]=zero_clip
    newpaircorrelation,_ = rdf_insertion_binned(coordinates,initial_guess,
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
                weights=[0 if i<2 else 1, 1/min([i,20])]
                )
        else:
             newpotential = np.exp(-pairpotential[-1])*pair_correlation_func/newpaircorrelation   

        newpotential[newpotential<zero_clip] = zero_clip
        newpotential = -np.log(newpotential)
        pairpotential.append(newpotential)
        
        #use it to calculate the new g(r)
        newpaircorrelation,c = rdf_insertion_binned(
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

def run_iterator_fitfunction(coordinates,pair_correlation_func,boundary,
                             potential_func,initial_guess=None,fit_bounds=None,
                             rmin=0,rmax=20,dr=0.5,convergence_tol=1e-5,
                             max_iterations=100,max_func_evals=100,
                             zero_clip=1e-20,**kwargs):
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
    rvals = np.arange(rmin,rmax+dr,dr)
    rcent = rvals[:-1]+dr/2
    
    #check inputs
    if len(pair_correlation_func) != len(rcent):
        raise ValueError('lenght pair_correlation_func does not match rmax and dr')
    
    if type(fit_bounds) == type(None):
        fit_bounds = (-np.inf,np.inf)
    
    #check dimensionality, select appropriate rdf_insertion routine
    boundary = np.array(boundary)
    if boundary.shape[-2] == 2:
        rdf_insertion_binned = rdf_insertion_binned_2d
    elif boundary.shape[-2] == 3:
        rdf_insertion_binned = rdf_insertion_binned_3d
    else:
        raise ValueError('data and boundaries must be 2- or 3-dimensional')
    
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
        newpaircorrelation,c = rdf_insertion_binned(coordinates, pairpotential,
                                                rmax, dr, boundary,rmin=rmin,
                                                insert_grid=True,
                                                **kwargs)
        
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


def run_iterator_fitfunction2(coordinates,pair_correlation_func,boundary,
                             potential_func,initial_guess=None,fit_bounds=None,
                             rmin=0,rmax=20,dr=0.5,convergence_tol=1e-5,
                             max_iterations=100,max_func_evals=100,
                             zero_clip=1e-20,**kwargs):
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
    rvals = np.arange(rmin,rmax+dr,dr)
    rcent = rvals[:-1]+dr/2
    
    #check inputs
    if len(pair_correlation_func) != len(rcent):
        raise ValueError('lenght pair_correlation_func does not match rmax and dr')
    
    if type(fit_bounds) == type(None):
        fit_bounds = (-np.inf,np.inf)
    
    #check dimensionality, select appropriate rdf_insertion routine
    boundary = np.array(boundary)
    if boundary.shape[-2] == 2:
        rdf_insertion_binned = rdf_insertion_binned_2d
    elif boundary.shape[-2] == 3:
        rdf_insertion_binned = rdf_insertion_binned_3d
    else:
        raise ValueError('data and boundaries must be 2- or 3-dimensional')
    
    #set input pair correlation to zero-clip to avoid devision by 0 errors
    pair_correlation_func[pair_correlation_func<zero_clip]=zero_clip
    
    #initialize lists to write results to upon fitfunction call (to get 
    #intermediate results from scipy.optimize.curve_fit)
    paircorrelations = []
    pairpotentials = []
    fit_params = []
    chi_squared = []
    counters = []
    
    def errorfunc(fitargs):
        
        if len(paircorrelations) > max_iterations:
            raise RuntimeError
        
        #calculate pairpotential function and call insertion routine
        pairpotential = lambda dist: potential_func(dist,*fitargs)
        newpaircorrelation,c = rdf_insertion_binned(coordinates, pairpotential,
                                                rmax, dr, boundary,rmin=rmin,
                                                insert_grid=True,
                                                **kwargs)
        
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
        print('iteration {:}, χ²={:4g}'.format(len(paircorrelations)-1,chi_squared[-1]))
        
        #return -np.log(newpaircorrelation)
        return error
    
    from scipy.optimize import minimize
    
    try:
        result = minimize(
            errorfunc,
            initial_guess,
            bounds = [bound for bound in zip(fit_bounds[0],fit_bounds[1])],
            options = {'disp':True}
        )
    except RuntimeError:
        print('max iterations reached')
    
    return chi_squared,pairpotentials,fit_params,paircorrelations,counters

def run_iterator_fitfunction3(coordinates,pair_correlation_func,boundary,
                             potential_func,initial_guess=None,fit_bounds=None,
                             rmin=0,rmax=20,dr=0.5,convergence_tol=1e-5,
                             max_iterations=100,max_func_evals=100,
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
    rvals = np.arange(rmin,rmax+dr,dr)
    rcent = rvals[:-1]+dr/2
    
    #check inputs
    if len(pair_correlation_func) != len(rcent):
        raise ValueError('lenght pair_correlation_func does not match rmax and dr')
    
    if type(initial_guess)==type(None):
        from inspect import signature
        initial_guess = [0]*(len(signature(potential_func).parameters)-1)
    
    if type(fit_bounds) == type(None):
        fit_bounds = (-np.inf,np.inf)
    
    #check dimensionality, select appropriate rdf_insertion routine
    boundary = np.array(boundary)
    if boundary.shape[-2] == 2:
        rdf_insertion_binned = rdf_insertion_binned_2d
    elif boundary.shape[-2] == 3:
        rdf_insertion_binned = rdf_insertion_binned_3d
    else:
        raise ValueError('data and boundaries must be 2- or 3-dimensional')
      
    #set up for 0th iteration seperately
    pairpotentials = [lambda r: potential_func(r,*initial_guess)]
    fitparams = [initial_guess]
    pair_correlation_func[pair_correlation_func<zero_clip]=zero_clip
    newpaircorrelation,_ = rdf_insertion_binned(coordinates,pairpotentials[-1],
                                                rmax,dr,boundary,rmin=rmin,
                                                **kwargs)
    newpaircorrelation[newpaircorrelation<zero_clip] = zero_clip#avoid deviding by zero
    paircorrelations = [newpaircorrelation]
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
                    np.exp(-pairpotentials[-1](rcent)),
                    np.exp(-pairpotentials[-1](rcent))*pair_correlation_func/newpaircorrelation
                ],
                axis=0,
                weights=[0 if i<2 else 1, 1/min([i,20])]
                )
        else:
             newpotential = np.exp(-pairpotentials[-1](rcent))*pair_correlation_func/newpaircorrelation

        newpotential[newpotential<zero_clip] = zero_clip
        newpotential = -np.log(newpotential)
        
        #fitting may fail if newpotential and fitfunction are very different,
        #or if the initial guess is far off.
        fitting_failed = True
        while fitting_failed:
            #try fitting and appending fitparameters
            try:
                fitparams.append(
                    curve_fit(
                        potential_func,
                        rcent,
                        newpotential,
                        bounds=fit_bounds,
                        sigma=1/np.sqrt(pair_correlation_func)
                    )[0]
                )
                fitting_failed=False
            
            #if fit is not succesful, use fitfunctionless method for updating and try again
            except RuntimeError:
                print('fitting failed, retrying...')
                newpaircorrelation,c = rdf_insertion_binned(
                    coordinates,
                    newpotential,
                    rmax,
                    dr,
                    boundary,
                    rmin=rmin,
                    **kwargs)
                newpotential = np.exp(-newpotential)*pair_correlation_func/newpaircorrelation
                newpotential[newpotential<zero_clip] = zero_clip
                newpotential = -np.log(newpotential)
        
        pairpotentials.append(lambda r: potential_func(r,*fitparams[-1]))
        
        #use it to calculate the new g(r)
        newpaircorrelation,c = rdf_insertion_binned(
            coordinates,
            pairpotentials[-1],
            rmax,
            dr,
            boundary,
            rmin=rmin,
            **kwargs)
        counters.append(c)
        newpaircorrelation[newpaircorrelation<1e-10] = 1e-10
        #newpaircorrelation[newpaircorrelation>20] = 20
        paircorrelations.append(newpaircorrelation)
        
        #calculate summed squared error
        chi_squared.append(np.mean((newpaircorrelation - pair_correlation_func)**2))
        
        print('\riteration {:}, χ²={:4g}'.format(i,chi_squared[-1]))
        if chi_squared[-1] < convergence_tol:
            break
        
        i += 1
    
    return chi_squared,pairpotentials,fitparams,paircorrelations,counters

def run_iterator_circularboundary(coordinates,pair_correlation_func,boundary_pos,
                              boundary_rad,initial_guess=None,rmin=0,rmax=20,
                              dr=0.5,convergence_tol=1e-5,max_iterations=100,
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
    boundary_pos : tuple of form `(y,x)`
        coordinates defining the center position of the circular boundary
    boundary_rad : float
        radius of the circular boundary
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
        Additional keyword arguments are passed on to 
        `rdf_insertion_binned_2d_circularboundary`

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
    
    References
    ----------
    [1] Stones, A. E., Dullens, R. P. A., & Aarts, D. G. A. L. (2019). Model-
    Free Measurement of the Pair Potential in Colloidal Fluids Using Optical 
    Microscopy. Physical Review Letters, 123(9), 098002. 
    https://doi.org/10.1103/PhysRevLett.123.098002
    
    See also
    --------
    rdf_insertion_binned_2d_circularboundaru : 2D routine for g(r) from 
    test-particle insertion under circular boundary conditions.
    """
    
    #create values for bin edges and centres of r
    rvals = np.arange(rmin,rmax+dr,dr)
    rcent = rvals[:-1]+dr/2
    
    #check inputs
    if len(pair_correlation_func) != len(rcent):
        raise ValueError('lenght pair_correlation_func does not match rmax and dr')
    
    if type(initial_guess)==type(None):
        initial_guess = np.zeros(len(rcent))
     
    #set up for 0th iteration seperately
    pairpotential = [initial_guess]
    pair_correlation_func[pair_correlation_func<zero_clip]=zero_clip
    newpaircorrelation,_ = rdf_insertion_binned_2d_circularboundary(
        coordinates,
        initial_guess,
        rmax,
        dr,
        boundary_pos,
        boundary_rad,
        rmin=rmin,
        **kwargs
    )
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
                weights=[0 if i<2 else 1, min([i,20])]
                )
        else:
             newpotential = np.exp(-pairpotential[-1])*pair_correlation_func/newpaircorrelation   

        newpotential[newpotential<zero_clip] = zero_clip
        newpotential = -np.log(newpotential)
        pairpotential.append(newpotential)
        
        #use it to calculate the new g(r)
        newpaircorrelation,c = rdf_insertion_binned_2d_circularboundary(
            coordinates,
            newpotential,
            rmax,
            dr,
            boundary_pos,
            boundary_rad,
            rmin=rmin,
            **kwargs
        )
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

#defs
def run_iterator_customboundary(coordinates,pair_correlation_func,boundary,
                                boundary_func,testparticle_func,
                                initial_guess=None,rmin=0,rmax=20,dr=0.5,
                                convergence_tol=1e-5,max_iterations=100,
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
    boundary_func : list of callable
        boundary correction function for each set in coordinates. Must take the
        following arguments (in this order):
            
            - list of bin edges
            - np.ndarray of coordinates
            - a single element from `boundary`
        
        and return a list of correction values between 0 and 1 which indicate 
        the fraction of the total circular/spherical shell area/volume around
        each particle in coordinates which is within the boundaries.
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
    
    References
    ----------
    [1] Stones, A. E., Dullens, R. P. A., & Aarts, D. G. A. L. (2019). Model-
    Free Measurement of the Pair Potential in Colloidal Fluids Using Optical 
    Microscopy. Physical Review Letters, 123(9), 098002. 
    https://doi.org/10.1103/PhysRevLett.123.098002
    
    See also
    --------
    rdf_insertion_binned_2d_customboundary : 2D routine for g(r) from test-
        particle insertion
    rdf_insertion_binned_3d : 3D routine for g(r) from test-particle insertion
    """
    
    #create values for bin edges and centres of r
    rvals = np.arange(rmin,rmax+dr,dr)
    rcent = rvals[:-1]+dr/2
    
    #check inputs
    if len(pair_correlation_func) != len(rcent):
        raise ValueError('lenght pair_correlation_func does not match rmax and dr')
    
    if type(initial_guess)==type(None):
        initial_guess = np.zeros(len(rcent))
    
    #check dimensionality, select appropriate rdf_insertion routine
    if coordinates[0].shape[1] == 2:
        rdf_insertion_binned = rdf_insertion_binned_2d_customboundary
    elif coordinates[0].shape[1] == 3:
        raise NotImplementedError()
    else:
        raise ValueError('data and boundaries must be 2- or 3-dimensional')
      
    #set up for 0th iteration seperately
    pairpotential = [initial_guess]
    pair_correlation_func[pair_correlation_func<zero_clip]=zero_clip
    newpaircorrelation,_ = rdf_insertion_binned(coordinates,initial_guess,
            rmax,dr,boundary,boundary_func,testparticle_func,rmin=rmin,**kwargs)
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
                weights=[0 if i<2 else 1, 1/min([i,20])]
                )
        else:
             newpotential = np.exp(-pairpotential[-1])*pair_correlation_func/newpaircorrelation   

        newpotential[newpotential<zero_clip] = zero_clip
        newpotential = -np.log(newpotential)
        pairpotential.append(newpotential)
        
        #use it to calculate the new g(r)
        newpaircorrelation,c = rdf_insertion_binned(
            coordinates,
            newpotential,
            rmax,
            dr,
            boundary,
            boundary_func,
            testparticle_func,
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