"""
Functions for determining the missing volume of a spherical shell which
falls partially outside of a cuboidal boundary.

Maarten Bransen, 2020
m.bransen@uu.nl
"""

#imports
import numpy as np
import numba as nb

#defs
def _sphere_shell_vol_fraction(r,boundary):
    """fully numpy vectorized function which returns the fraction of the volume 
    of spherical shells r+dr around particles, for a list of particles and a 
    list of bin-edges for the radii simultaneously. Analytical functions for 
    calculating the intersection volume of a sphere and a coboid are taken from
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

@nb.njit()
def _sphere_oct_vol(r,hx,hy,hz):
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
def _sphere_shell_vol_fraction_nb(radii,boundary):
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
                        vol[i,j] += _sphere_oct_vol(r,abs(hx),abs(hy),abs(hz))
    
    part_shell = vol[:,1:] - vol[:,:-1]
    tot_shell = 4/3*np.pi * (radii[1:]**3 - radii[:-1]**3)
    
    return part_shell / tot_shell

def _sphere_shell_vol_frac_periodic(r,boxsize):
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
