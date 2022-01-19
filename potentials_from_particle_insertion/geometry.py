"""
Functions for determining the missing volume of a circular (2D) or spherical 
(3D)shell which falls partially outside of a boundary of given shape.

Maarten Bransen, 2020
m.bransen@uu.nl
"""

#imports
import numpy as np
from .distribution_functions import _numba_available
if _numba_available:
    import numba as nb

#defs
def _sphere_shell_vol_frac_in_cuboid(r,boundary):
    """fully numpy vectorized function which returns the fraction of the volume 
    of spherical shells r+dr around particles, for a list of particles and a 
    list of bin-edges for the radii simultaneously. Analytical functions for 
    calculating the intersection volume of a sphere and a cuboid are taken from
    ref. [1].

    Parameters
    ----------
    r : numpy.array
        list of edges for the bins in r, where the number of shells is len(r)-1
    boundary : numpy.array of shape n*3*2
        list of boundary values shifted with respect to the particle 
        coordinates such that the particles are in the origin, in other words
        the distances to all 6 boundaries. First dimension contains all 
        particles, second dimension refers to spatial dimension (z,y,x) and
        third dimension is used to split the boundaries in the negative and 
        positive directions (or min and max of bounding box in each dimension)

    Returns
    -------
    numpy.array of shape n*(len(r)-1)
        array containing the fraction of each shell in r around each particle
        which lies inside of the boundaries, i.e. v_in/v_tot
    
    References
    ----------
    [1] Kopera, B. A. F., & Retsch, M. (2018). Computing the 3D Radial 
    Distribution Function from Particle Positions: An Advanced Analytic 
    Approach. Analytical Chemistry, 90(23), 13909–13914. 
    https://doi.org/10.1021/acs.analchem.8b03157
    
    See also
    --------
    _sphere_shell_vol_fraction_nb, a numba-compiled version of this function.
    """
    #initialize array with row for each particle and column for each r
    nrow,ncol = len(boundary),len(r)
    vol = np.zeros((nrow,ncol))
    
    #mirror all particle-wall distances into positive octant
    boundary = abs(boundary)
    
    #loop over all sets of three adjecent boundaries
    for hz in (boundary[:,0,0],boundary[:,0,1]):
        for hy in (boundary[:,1,0],boundary[:,1,1]):
            for hx in (boundary[:,2,0],boundary[:,2,1]):
                
                #if box octant entirely inside of sphere octant, add box oct volume
                boxmask = (hx**2+hy**2+hz**2)[:,np.newaxis] < r**2
                vol[boxmask] += np.broadcast_to((hx*hy*hz)[:,np.newaxis],(nrow,ncol))[boxmask]
                
                #to the rest add full sphere octant (or 1/8 sphere)
                boxmask = ~boxmask
                vol[boxmask] += np.broadcast_to((np.pi/6*r**3)[np.newaxis,:],(nrow,ncol))[boxmask]
                
                #remove hemispherical caps
                for h in (hz,hy,hx):
                    
                    #check where to change values, select those items
                    mask = (h[:,np.newaxis] < r)*boxmask
                    indices = np.where(mask)
                    
                    h = h[indices[0]]
                    rs = r[indices[1]]
                    
                    #subtract cap volume
                    vol[mask] -= np.pi/4*(2/3*rs**3-h*rs**2+h**3/3)
                
                #loop over over edges and add back doubly counted edge pieces
                for h0,h1 in ((hz,hy),(hz,hx),(hy,hx)):
                    
                    #check where to change values, select those values
                    mask = (h0[:,np.newaxis]**2+h1[:,np.newaxis]**2 < r**2)*boxmask
                    indices = np.where(mask)

                    h0 = h0[indices[0]]
                    h1 = h1[indices[0]]
                    rs = r[indices[1]]
                    
                    #add back edge wedges
                    c = np.sqrt(rs**2-h0**2-h1**2)
                    vol[indices] += rs**3/6*(np.pi-2*np.arctan(h0*h1/rs/c)) +\
                        (np.arctan(h0/c)-np.pi/2)*(rs**2*h1-h1**3/3)/2 +\
                        (np.arctan(h1/c)-np.pi/2)*(rs**2*h0-h0**3/3)/2 +\
                        h0*h1*c/3
    
    #calculate each shell by subtracting the sphere volumes of previous r
    part_shell = vol[:,1:] - vol[:,:-1]
    tot_shell = 4/3*np.pi * (r[1:]**3 - r[:-1]**3)
    
    return part_shell/tot_shell

def _sphere_shell_vol_frac_in_cuboid_periodic(r,boxsize):
    """returns effective volume of each shell defined by the intervals between
    the radii in r, under periodic boundary conditions in a cubic box with 
    edge length boxsize. Effective volume means the volume to which no shorter
    path exists through the periodic boundaries than the r under consideration.
    
    Analytical formulas taken from ref. [1].

    Parameters
    ----------
    r : numpy.array of float
        list of bin edges defining the shells where each interval between 
        values in r is treated as shell.
    boxsize : float
        edge length of the cubic box.

    Returns
    -------
    numpy.array of float
        Effective volume for each interval r -> r+dr in r. Values beyond 
        sqrt(3)/2 boxsize are padded with numpy.nan values.
    
    References
    ----------
    [1] Markus Seserno (2014). How to calculate a three-dimensional g(r) under
    periodic boundary conditions.
    https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf

    """
    #init volume list, scale r to boxsize
    vol = np.zeros(len(r))
    r = r/boxsize
    
    #up to half boxlen, normal sphere vol
    mask = r <= 1/2
    vol[mask] = 4/3*np.pi * r[mask]**3
    
    #between boxlen/2 and sqrt(2)/2 boxlen
    mask = (1/2 < r) & (r <= np.sqrt(2)/2)
    vol[mask] = -np.pi/12 * (3 - 36*r[mask]**2 + 32*r[mask]**3)
    
    #between sqrt(2)/2 boxlen and sqrt(3)/2 boxlen
    mask = (np.sqrt(2)/2 < r) & (r <= np.sqrt(3)/2)
    vol[mask] = -np.pi/4 + 3*np.pi*r[mask]**2 + np.sqrt(4*r[mask]**2 - 2) \
        + (1 - 12*r[mask]**2) * np.arctan(np.sqrt(4*r[mask]**2 - 2)) \
        + 2/3 * r[mask]**2 * 8*r[mask] *np.arctan(
            2*r[mask]*(4*r[mask]**2 - 3) / (np.sqrt(4*r[mask]**2 - 2)*(4*r[mask]**2 + 1))
            )
    
    #beyond sqrt(3)/2 boxlen there is no useful info
    vol[np.sqrt(3)/2 < r] = np.nan
    
    part = vol[1:] - vol[:-1]
    full = 4/3*np.pi*(r[1:]**3 - r[:-1]**3)
    
    return part/full

def _sphere_shell_vol_frac_in_sphere(r,d,boundaryrad):
    """fully analytical and numpy vectorized function which returns the 
    fraction of the volume of spherical shells r+dr around particles which are 
    within the spherical boundary of radius boundaryrad.
    
    Parameters
    ----------
    r : numpy.array
        list of edges for the bins in r, where the number of shells is len(r)-1
    d : numpy.array of length n
        list of distances between the particles and the origin of the spherical
        boundary, for each particle.
    boundaryrad : float
        radius defining the circular boundary

    Returns
    -------
    numpy.array of shape n*(len(r)-1)
        array containing the fraction of each spherical shell in r around each 
        particle which lies inside of the boundaries, i.e. V_in/Vtot
    """
    #https://en.wikipedia.org/wiki/Spherical_cap#Volumes_of_union_and_intersection_of_two_intersecting_spheres
    
    nrow,ncol = len(d),len(r)
    
    #init list and array for full and partial sphere shells
    full =  4*np.pi*r**3 / 3
    part = np.zeros((nrow,ncol))
    
    #broadcast to correct shape
    d = np.broadcast_to(d[:,np.newaxis],(nrow,ncol))
    r = np.broadcast_to(r[np.newaxis,:],(nrow,ncol))
    
    #find spheres that intersect the boundary
    mask = d + r > boundaryrad
    part[mask] = \
        np.pi/(12*d[mask]) * (r[mask]+boundaryrad-d[mask])**2 * \
            (d[mask]**2 + 2*d[mask]*(r[mask]+boundaryrad) - \
             3*(r[mask]-boundaryrad)**2)
    
    #rest has full volume
    part[~mask] += np.broadcast_to(full[np.newaxis,:],(nrow,ncol))[~mask]
    
    #convert spheres to shells
    part = part[:,1:] - part[:,:-1]
    full = full[1:] - full[:-1]
    
    return part/full
    
#@nb.njit
def _circle_ring_area_frac_in_rectangle(r,boundary):
    """fully numpy vectorized function which returns the fraction of the area 
    of circular rings r+dr around particles, for a list of particles and a 
    list of bin-edges for the radii simultaneously. Uses analytical formulas
    for the intersection area of a circle and a rectangle.
    
    Parameters
    ----------
    r : numpy.array
        list of edges for the bins in r, where the number of shells is len(r)-1
    boundary : numpy.array of shape n*2*2
        list of boundary values shifted with respect to the particle 
        coordinates such that the particles are in the origin, in other words
        the distances to all 6 boundaries. First dimension contains all 
        particles, second dimension refers to spatial dimension (y,x) and
        third dimension is used to split the boundaries in the negative and 
        positive directions (or min and max of bounding box in each dimension)

    Returns
    -------
    numpy.array of shape n*(len(r)-1)
        array containing the fraction of each circle ring in r around each 
        particle which lies inside of the boundaries, i.e. A_in/A_tot
    """
    
    #initialize array with row for each particle and column for each r
    nrow,ncol = len(boundary),len(r)
    area = np.zeros((nrow,ncol),dtype=float)
    
    #mirror all particle-edge distances into positive octant
    boundary = abs(boundary)
    
    #loop over all quarters
    for hy in (boundary[:,0,0],boundary[:,0,1]):
        for hx in (boundary[:,1,0],boundary[:,1,1]):
            
            #if circle edge entirely out of boxquarter, add boxquarter area
            boxmask = (hx**2+hy**2)[:,np.newaxis] < r**2
            area[boxmask] += np.broadcast_to((hx*hy)[:,np.newaxis],(nrow,ncol))[boxmask]
            
            #to the rest add a quarter sphere
            boxmask = ~boxmask
            area[boxmask] += np.broadcast_to((np.pi/4*r**2)[np.newaxis,:],(nrow,ncol))[boxmask]
            
            #remove hemispherical caps
            for h in (hy,hx):
                
                #check where to change values, select those items
                mask = (h[:,np.newaxis] < r)*boxmask
                indices = np.where(mask)
                
                h = h[indices[0]]
                rs = r[indices[1]]
                
                #subtract cap area
                area[mask] -= rs*(rs*np.arccos(h/rs) - h*np.sqrt(1-h**2/rs**2))/2
    
    #calculate each shell by subtracting the sphere volumes of previous r
    part_ring = area[:,1:] - area[:,:-1]
    tot_ring = np.pi * (r[1:]**2 - r[:-1]**2)
    
    return part_ring/tot_ring

def _circle_ring_area_frac_in_rectangle_periodic(r,boxsize):
    """returns effective area of each ring defined by the intervals between
    the radii in r, under periodic boundary conditions in a square box with 
    edge length boxsize. Effective area means the area to which no shorter
    path exists through the periodic boundaries than the r under consideration.
    
    Parameters
    ----------
    r : numpy.array of float
        list of bin edges defining the shells where each interval between 
        values in r is treated as shell.
    boxsize : float
        edge length of the square box.

    Returns
    -------
    numpy.array of float
        Effective volume for each interval r -> r+dr in r. Values beyond 
        sqrt(2)/2 boxsize are padded with numpy.nan values.

    """
    
    #scale r to boxsize
    r = r/boxsize
    
    #add full circle area to each
    area = np.pi*r**2
    
    #between boxlen/2 and sqrt(2)/2 boxlen subtract 4 circle segments
    mask = (1/2 < r) & (r <= np.sqrt(2)/2)
    area[mask] -= 4*r[mask]*(r[mask]*np.arccos(1/(2*r[mask])) - np.sqrt(1-1/(4*r[mask]**2))/2)
    
    #beyond sqrt(2)/2 boxlen there is no useful info
    area[np.sqrt(2)/2 < r] = np.nan
    
    part_ring = area[1:] - area[:-1]
    full_ring = np.pi*(r[1:]**2 - r[:-1]**2)
    
    return part_ring/full_ring

def _circle_ring_area_frac_in_circle(r,distances,boundaryrad):
    """fully analytical and numpy vectorized function which returns the 
    fraction of the area of circular rings r+dr around particles which is 
    within the circular boundary conditions.
    
    Parameters
    ----------
    r : numpy.array
        list of edges for the bins in r, where the number of shells is len(r)-1
    distances : numpy.array of length n
        list of distances between the particles and the center of the circular
        boundary, for each particle.
    boundaryrad : float
        radius defining the circular boundary

    Returns
    -------
    numpy.array of shape n*(len(r)-1)
        array containing the fraction of each circle ring in r around each 
        particle which lies inside of the boundaries, i.e. A_in/A_tot
    """
    #https://diego.assencio.com/?index=8d6ca3d82151bad815f78addf9b5c1c6
    #note that I have verified it does not matter to switch r1 and r2
    
    #initialize zeros array with row for each particle and column for each r
    nrow,ncol = len(distances),len(r)
    area = np.zeros((nrow,ncol),dtype=float)
    
    #check that no particles are outside boundary
    if any(distances>boundaryrad):
        raise ValueError('all particles must be within boundary')
    
    #circles entirely enclosed in boundary
    mask1 = (r<=boundaryrad)[np.newaxis,:] & (distances[:,np.newaxis] <=  (boundaryrad - r)[np.newaxis,:])
    area[mask1] += np.broadcast_to((np.pi*r**2)[np.newaxis,:],(nrow,ncol))[mask1]
   
    #boundary entirely enclosed in circle
    mask2 = (r>boundaryrad)[np.newaxis,:] & (distances[:,np.newaxis] <=  (r - boundaryrad)[np.newaxis,:])
    area[mask2] += np.pi*boundaryrad**2
    
    #all other places, calculate intersection
    mask1 = ~np.logical_or(mask1,mask2)

    r1 = boundaryrad
    r2 = np.broadcast_to(r[np.newaxis,:],(nrow,ncol))[mask1]
    d = np.broadcast_to(distances[:,np.newaxis],(nrow,ncol))[mask1]
    d1 = (r1**2-r2**2+d**2)/(2*d)
    d2 = d - d1
    
    area[mask1] += \
        r1**2 * np.arccos(d1/r1) - d1 * np.sqrt(r1**2-d1**2) +\
        r2**2 * np.arccos(d2/r2) - d2 * np.sqrt(r2**2-d2**2)
    
    #calculate each shell by subtracting the sphere volumes of previous r
    part_ring = area[:,1:] - area[:,:-1]
    tot_ring = np.pi * (r[1:]**2 - r[:-1]**2)
    
    return part_ring/tot_ring

if _numba_available:
    @nb.njit()
    def _sphere_oct_vol_nb(r,hx,hy,hz):
        """numba-compiled subroutine for calculating volume of a sphere octant 
        intersecting with one boundary octant. See _sphere_shell_vol_fraction_nb.
        """
        
        #sphere surface entirely outside of box, return box oct vol
        if hx**2+hy**2+hz**2 < r**2:
            return hx*hy*hz
        
        #total sphere oct vol
        v = np.pi/6*r**3
        
        #remove spherical caps
        for h in (hx,hy,hz):
            if h < r:
                v -= np.pi/4*(2/3*r**3 - h*r**2 + h**3/3)
            
        #add back edge wedges where two caps overlap
        for h0,h1 in [(hx,hy),(hx,hz),(hy,hz)]:
            if h0**2+h1**2 < r**2:
                
                c = np.sqrt(r**2-h0**2-h1**2)
                v += r**3*(np.pi-2*np.arctan(h0*h1/(r*c)))/6 +\
                    (np.arctan(h0/c)-np.pi/2)*(h1*r**2-h1**3/3)/2 +\
                    (np.arctan(h1/c)-np.pi/2)*(h0*r**2-h0**3/3)/2 +\
                    h0 * h1 * c / 3
        return v

    @nb.njit()
    def _sphere_shell_vol_frac_in_cuboid_nb(radii,boundary):
        """numba-compiled version of _sphere_shell_vol_fraction, with identical
        input and output but different computational details.
        
        For usage details see _sphere_shell_vol_fraction()"""
        #initialize array with row for each particle and column for each r
        nrow,ncol = len(boundary),len(radii)
        vol = np.zeros((nrow,ncol))
        
        for i in nb.prange(nrow):
            p = boundary[i]
            for j in range(ncol):
                r = radii[j]
                if r<1e-20:#skip radii of 0 (within 1e-20 tolerance)
                    continue
                for hz in p[0]:
                    for hy in p[1]:
                        for hx in p[2]:
                            vol[i,j] += _sphere_oct_vol_nb(r,abs(hx),abs(hy),abs(hz))
        
        part_shell = vol[:,1:] - vol[:,:-1]
        tot_shell = 4/3*np.pi * (radii[1:]**3 - radii[:-1]**3)
        
        return part_shell / tot_shell