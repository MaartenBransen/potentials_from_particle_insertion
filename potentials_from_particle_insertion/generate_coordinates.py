"""
This file contains funtions for generating pseudorandom coordinates in 
differently shaped boundary conditions

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

import numpy as np
from scipy.spatial import cKDTree

def _rand_coord_in_box(boundary,n=1):
    """
    generate n random coordinates uniformly distributed in the box defined by
    boundary

    Parameters
    ----------
    boundary : np.array of form ((zmin,zmax),(ymin,ymax),(xmin,xmax))
        The half-open interval in which to generate the coordinates, can have
        any number of dimensions
    n : int, optional
        The number of coordinates to generate. The default is 1.
        
    Returns
    -------
    numpy.array of n*ndim
        Randomly generated coordinates in box of ndim dimension

    """
    return np.random.random_sample((n,len(boundary)))*\
        (boundary[:,1]-boundary[:,0])[np.newaxis,:]+boundary[np.newaxis,:,0]
        
def _rand_coord_in_circle(boundary_pos,boundary_rad,n=1):
    """
    generate n random coordinates uniformly distributed in the box defined by
    boundary

    Parameters
    ----------
    boundary_pos : tuple of form (y,x)
        The position of the centre of the circle in which to generate 
        coordinates
    boundary_rad : float
        the radius of the circle in which to generate coordinates
    n : int, optional
        The number of coordinates to generate. The default is 1.
        
    Returns
    -------
    numpy.array of n*2
        Randomly generated 2D coordinates within the circle

    References
    ----------
    https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/50746409#50746409
    """
    r = boundary_rad*np.sqrt(np.random.random_sample(n))
    theta = 2*np.pi*np.random.random_sample(n)
    coords = np.empty((n,2))
    coords[:,0] = boundary_pos[0] + r*np.sin(theta)
    coords[:,1] = boundary_pos[1] + r*np.cos(theta)
    
    return coords

def _rand_coord_in_sphere(boundary_pos,boundary_rad,n=1):
    """
    generate n random coordinates uniformly distributed in a spherical box

    Parameters
    ----------
    boundary_pos : tuple of form (z,y,x)
        The position of the centre of the circle in which to generate 
        coordinates
    boundary_rad : float
        the radius of the circle in which to generate coordinates
    n : int, optional
        The number of coordinates to generate. The default is 1.
        
    Returns
    -------
    numpy.array of n*3
        Randomly generated 2D coordinates within the circle

    References
    ----------
    https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
    """
    r = boundary_rad*np.cbrt(np.random.random_sample(n))
    theta = 2*np.pi*np.random.random_sample(n)
    phi = np.arccos(2*np.random.random_sample(n)-1)
    
    coords = np.empty((n,3))
    coords[:,0] = boundary_pos[0] + r * np.cos(phi)
    coords[:,1] = boundary_pos[1] + r * np.sin(phi) * np.sin(theta)
    coords[:,2] = boundary_pos[2] + r * np.sin(phi) * np.cos(theta)
    
    return coords

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
        coord[rejects] = np.random.random_sample((rejects.sum(),len(boundary))) \
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
    [1] Marsaglia, G. "Choosing a Point from the Surface of a Sphere." Ann. 
    Math. Stat. 43, 645-646, 1972
    
    [2] Muller, M. E. "A Note on a Method for Generating Points Uniformly on
    N-Dimensional Spheres." Comm. Assoc. Comput. Mach. 2, 19-20, Apr. 1959.


    """
    vec = np.random.randn(3, npoints)
    vec = radius * vec / np.linalg.norm(vec, axis=0)
    coord = np.array([list(origin)]*npoints)+vec.T
    return coord

def _coord_grid_in_box(boundary,n=1):
    """
    generate approximately n coordinates on a uniformly distributed grid in the
    box defined by boundary, excluding values at the boundaries. n is 
    approximate since an integer number of equally sized steps along each 
    dimension is needed

    Parameters
    ----------
    boundary : np.array of form ((zmin,zmax),(ymin,ymax),(xmin,xmax))
        The half-open interval in which to generate the coordinates, can have
        any number of dimensions
    n : int, optional
        The number of coordinates to generate. The default is 1.
        
    Returns
    -------
    numpy.array of n*ndim
        Randomly generated coordinates in box of ndim dimension

    """
    edge_lengths = boundary[:,1]-boundary[:,0]
    n_per_dim = np.round(n**(1/3)*edge_lengths/np.prod(edge_lengths)**(1/3)).astype(np.uint8)
    n_per_dim[n_per_dim<1] = 1
    steps = [np.linspace(dmin,dmax,dn+1,endpoint=False)[1:] 
             for dmin,dmax,dn in zip(boundary[:,0],boundary[:,1],n_per_dim)]
    return np.transpose([dim.ravel() for dim in np.meshgrid(*steps)])
    
    
