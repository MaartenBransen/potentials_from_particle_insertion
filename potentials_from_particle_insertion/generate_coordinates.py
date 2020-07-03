import numpy as np
from scipy.spatial import cKDTree

def _rand_coord_in_box(boundary,n=1):
    """
    generate n random coordinates uniformly distributed in the box defined by
    boundary

    Parameters
    ----------
    boundary : iterable of form ((zmin,zmax),(ymin,ymax),(xmin,xmax))
        The half-open interval in which to generate the coordinates.
    n : int, optional
        The number of coordinates to generate. The default is 1.
        
    Returns
    -------
    numpy.array of n*3
        Randomly generated coordinates in box

    """
    return np.random.random_sample((n,3))*(boundary[:,1]-boundary[:,0])[np.newaxis,:]+boundary[np.newaxis,:,0]

def _rand_coord_at_dist(boundary,coordinates,rmin,n=1,timeout=100):
    """Random coordinate from contineous uniform distribution of box given by
    half-open intervals in boundary, where each coordinate is at least `rmin` 
    away from any point in `coordinates`. Coordinates are generated randomly
    in the box, checked for proximity, and replaced if needed in a loop that 
    times out after `timeout` iterations.

    Parameters
    ----------
    boundary : iterable of form ((zmin,zmax),(ymin,ymax),(xmin,xmax))
        The half-open interval in which to generate the coordinates.
    coordinates : 2D numpy.array with [z,y,x] in second dimension
        Set coordinates that generated coordinates must be away from
    rmin : float
        Minimum distance between input coordinates and generated coordinates
    n : int, optional
        The number of coordinates to generate. The default is 1.
    timeout : int, optional
        The maximum number of attempts to find an insertion location for 
        each particle before giving up. The default is 100.

    Returns
    -------
    numpy.array of n*3
        Randomly generated coordinates which are at least rmin away from input

    """    
    coord = _rand_coord_in_box(boundary,n=n)
    if rmin == 0:
        return coord
    
    #initialize cKDTree of coordinates to stay away from
    tree = cKDTree(coordinates)
    
    #find distances closer than rmin
    dist,_ = tree.query(coord,k=1,distance_upper_bound=rmin)
    rejects = np.isfinite(dist)
    
    i=0
    #iterate until there are no more rejected particles
    while rejects.sum()>0:
        if i>=timeout:#end if it takes too many trials
            print('coordinate placement timed out, continuing with {:} coordinates'.format(sum(~rejects)))
            return coord[~rejects]
        
        #replace rejected values
        coord[rejects] = np.random.random_sample((rejects.sum(),3)) \
            *(boundary[:,1]-boundary[:,0])[np.newaxis,:]+boundary[np.newaxis,:,0]
        
        #check replaced values, and update list of rejects
        dist,_ = tree.query(coord[rejects],k=1,distance_upper_bound=rmin)
        rejects[rejects] = np.isfinite(dist)
        
        i+= 1
    
    return coord

def _rand_coord_on_sphere(npoints=1,radius=1,origin=[0,0,0]):
    """
    generate `npoints` random coordinates on the surface of a sphere of radius
    `radius` around the center point `origin`.
    
    Works by taking a vector of three normally distributed random variables
    around 0, normalizing the vector and multiplying by `radius`
    
    Parameters
    ----------
    npoints : int, optional
        number of points to generate. The default is 1.
    radius : float, optional
        distance from origin at which to generate coordinates, or the radius of
        the sphere on which the generated points lie. The default is 1.
    origin : list of floats of form [z,y,x], optional
        the coordinates for the centrepoint of the sphere on which to generate
        positions. The default is [0,0,0].

    Returns
    -------
    coord : array of floats with shape npoints x 3
        The list of generated coordinates.
        
    References
    ----------
    Marsaglia, G. "Choosing a Point from the Surface of a Sphere." Ann. Math. 
        Stat. 43, 645-646, 1972
    Muller, M. E. "A Note on a Method for Generating Points Uniformly on
        N-Dimensional Spheres." Comm. Assoc. Comput. Mach. 2, 19-20, Apr. 1959.


    """
    vec = np.random.randn(3, npoints)
    vec = radius * vec / np.linalg.norm(vec, axis=0)
    coord = np.array([list(origin)]*npoints)+vec.T
    return coord