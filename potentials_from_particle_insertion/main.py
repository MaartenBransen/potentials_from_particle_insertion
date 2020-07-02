# -*- coding: utf-8 -*-
"""
Set of functions for determining the missing volume of a spherical shell which
falls partially outside of a cuboidal boundary.

Maarten Bransen, 2020
m.bransen@uu.nl
"""

#%% imports

import numpy as np
import numba as nb
import sys
import pandas as pd
from matplotlib import pyplot as plt
from glob import glob
import trackpy as tp
from matplotlib.cm import get_cmap
from scipy.spatial import cKDTree


#%% defs




def run_iteration(coordinates,pair_correlation_func,boundary,
                  initial_guess=None,rmin=0,rmax=20,dr=0.5,convergence_tol=0.1,
                  max_iterations=100,zero_clip=1e-10,**kwargs):
    """
    Run the algorithm to solve for the pairwise potential that most accurately
    reproduces the radial distribution function using test-particle insertion

    Parameters
    ----------
    coordinates : list of numpy.array
        List of sets of coordinates, where each item along the 0th dimension is
        a n*3 list of particle coordinates in `[z,y,x]`, e.g. one z-stack. Each
        set of coordinates is not required to have the same number of particles
        but all stacks must share the same bounding box as given by boundary.
    pair_correlation_func : list of float
        bin values for the true pair correlation function that the algorithm 
        will try to match. 
    boundary : tuple of form `((zmin,zmax),(ymin,ymax),(xmin,xmax))`
        The min and max positions along each dimension defining the bounding 
        box in which particles can exist.
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
    **kwargs : key=value
        Additional keyword arguments are passed on to
        `pair_correlation_boundary_3d`

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
    pair_correlation_boundary_3d

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
    newpaircorrelation,_ = pair_correlation_boundary_3d_new(coordinates,initial_guess,rmax,dr,boundary,rmin=rmin,**kwargs)
    newpaircorrelation[newpaircorrelation<zero_clip] = zero_clip#avoid deviding by zero
    paircorrelation = [newpaircorrelation]
    #newpaircorrelation[newpaircorrelation>20] = 20
    chi_squared = [np.mean((newpaircorrelation - pair_correlation_func)**2)]
    print('\riteration 0, χ²={:4g}'.format(chi_squared[-1]))
    
    #start the main iterative loop
    i = 1
    counters = []
    while i < max_iterations:
        
        #calculate the new pairwise potential, with relaxation
        #newpotential = np.mean([np.exp(-pairpotential[-1]),np.exp(-pairpotential[-1])*pair_correlation_func/newpaircorrelation],axis=0)
        newpotential = np.average(
            [
                np.exp(-pairpotential[-1]),
                np.exp(-pairpotential[-1])*pair_correlation_func/newpaircorrelation
            ],
            axis=0,
            weights=[i,max_iterations-i]
            )

        newpotential[newpotential<zero_clip] = zero_clip
        newpotential = -np.log(newpotential)
        pairpotential.append(newpotential)
        
        #use it to calculate the new g(r)
        newpaircorrelation,c = pair_correlation_boundary_3d_new(
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
    

#%% temp defs

def dumpreader_pandas(filename,scaled_box=False,periodic_boundary=False):
    """
    reads in a list of dumpfiles as provided by Ian Jenkins
    """

    #read in file and strip '\n' off ends
    with open(filename) as file:
        filedata = [line[:-1] for line in file.readlines()]
    
    #get header info
    t = float(filedata[1])
    n = int(filedata[3])
    bb = [[float(b) for b in line.split()] for line in filedata[5:8]]
    
    #gen dataframe of coordinates
    df = pd.DataFrame(
            columns=['id','x','y','z'],
            data=[line.split() for line in filedata[9:n+9]]
            )
    
    #set correct data types
    df = df.astype({'id':int,'x':float,'y':float,'z':float},copy=False)
    df = df.set_index('id')

    df = df[
            (df.x >= bb[0][0]) & (df.x < bb[0][1]) &
            (df.y >= bb[1][0]) & (df.y < bb[1][1]) &
            (df.z >= bb[2][0]) & (df.z < bb[2][1])
            ]

    #optionally rescale coordinates to dimensions of box
    if scaled_box:
        df['x'] = bb[0][0] + df['x']*(bb[0][1]-bb[0][0])
        df['y'] = bb[1][0] + df['y']*(bb[1][1]-bb[1][0])
        df['z'] = bb[2][0] + df['z']*(bb[2][1]-bb[2][0])
    
    #modulo all coordinates within periodic boundary conditions
    if periodic_boundary:
        df['x'] = (df['x']-bb[0][0]) % (bb[0][1]-bb[0][0]) + bb[0][0]
        df['y'] = (df['y']-bb[1][0]) % (bb[1][1]-bb[1][0]) + bb[1][0]
        df['z'] = (df['z']-bb[2][0]) % (bb[2][1]-bb[2][0]) + bb[2][0]
    
    return df,bb,t

from scipy.spatial import cKDTree
def pair_correlation_3d_periodic(coords,rmax,boundary,rmin=0,dr=None,query_indices=None,handle_edge=True):
    """
    3D pair correlation under periodic boundary conditions in a cubic box
    
    References
    ----------
    [1] Markus Seserno (2014). How to calculate a three-dimensional g(r) under
    periodic boundary conditions.
    https://www.cmu.edu/biolphys/deserno/pdf/gr_periodic.pdf

    """
    
    #check inputs
    boundary = np.array(boundary)
    
    if type(dr)==type(None):
        dr = rmax/20
        
    if type(query_indices)==type(None):
        query_indices = coords.index
        
    if len(set(boundary[:,1]-boundary[:,0]))==1:#if cubic box
        if rmax > 0.5*np.sqrt(3)*(boundary[0,1]-boundary[0,0]) and handle_edge:#max distance is 0.5*sqrt(3)*L
            raise ValueError('rmax cannot be longer than sqrt(3)/2 the box size')
    else:
        if rmax > 0.5*min(boundary[:,1]-boundary[:,0]) and handle_edge:
            raise NotImplementedError('rmax larger than half the smallest box '+
                                      'dimension is not implemented for non-cubic box')
        
    #correct coords for boundary such that all mins are set to 0
    querycoords = coords.loc[query_indices][['x','y','z']].to_numpy()
    coords = coords[['x','y','z']].to_numpy()
    coords -= boundary[:,0]
    boundary[:,1] -= boundary[:,0]
    
    #calculate bins and density
    bins = np.arange(rmin,rmax+dr,dr)
    density = (len(coords)-1)/np.product(boundary[:,1])#-1 for reference particle in origin
    
    #init and query tree
    tree = cKDTree(coords,boxsize=boundary[:,1])
    dist,indices = tree.query(querycoords,k=len(coords),distance_upper_bound=rmax)
    mask = (~np.isfinite(dist)) | (dist<=rmin)#remove pairs with self, smaller than rmin, and np.inf fill values
    
    if handle_edge and len(set(boundary[:,1]))==1:#cubic box
        L = boundary[0,1]
        
        #define function p from ref [1] which is normalized to L
        def p(r):
            if r <= 1/2:#between 0 and L/2 do normal correction
                return 4*np.pi*r**2
            if r <= np.sqrt(2)/2:#between L/2 and sqrt(2)*L/2
                return 2*np.pi*r*(3-4*r)*dr
            #between L*sqrt(2)/2 and L*sqrt(3)/2
            f1 = np.arctan(np.sqrt(4*r**2 - 2))
            f2 = 8*r*np.arctan(2*r*(4*r**2-3)/(np.sqrt(4*r**2-2)*(4*r**2+1)))
            return 2*r*(3*np.pi - 12*f1 + f2)
        
        correction = [len(querycoords)*density*p(r/L)*L**2*dr for r in (bins[1:]+bins[:-1])/2]
    
    else:
        correction = [len(querycoords)*density*4*np.pi*r**2*dr for r in (bins[1:]+bins[:-1])/2]
    
    counts = np.histogram(dist[~mask],bins=bins)[0]
    rdf = counts/correction
    
    return bins,rdf,counts
    
def u_lj(x,eps=1):
    return 4*eps*(x**(-12)-x**(-6))

def f_lj(x,eps=1,cutoff=5,smooth=0.5):
    if x <= cutoff-smooth:
        return 4*eps*(12*x**(-13) - 6*x**(-7))
    elif x < cutoff:
        a = (0 - 4*eps*(12*(cutoff-smooth)**(-13)- 6*(cutoff-smooth)**(-7)))/smooth
        b = a*cutoff
        return a*x-b
    else:
        return 0

def u_yk(x,eps=1,debye=1,cutoff=10,smooth=0.5):
    if x <= cutoff-smooth:
        return eps * np.exp(-x/debye)/x
    elif x <= cutoff:
        a = (0 - eps*np.exp(-(cutoff-smooth)/debye)/(cutoff-smooth))/smooth
        b = a*cutoff
        return a*x-b
    else:
        return 0
    
def f_yk(x,eps=1,debye=1,cutoff=10,smooth=0.5):
    
    if x <= cutoff-smooth:
        res = eps * np.exp(-x/debye) * (1/(x*debye) + 1/x**2)
    elif x <= cutoff:
        a = (0 - eps*np.exp(-(cutoff-smooth)/debye)* (1/((cutoff-smooth)*debye) + 1/(cutoff-smooth)**2))/smooth
        b = a*cutoff
        res = a*x-b
    else:
        res = 0
        
    return min(res,10)


params = {
    'figure.autolayout' : True,
    'xtick.top' : False,
    'xtick.bottom' : True,
    'xtick.minor.visible' : True,
    'xtick.labelsize':'medium',
    'ytick.left' : True,
    'ytick.right' : False,
    'ytick.minor.visible' : True,
    'figure.figsize' : (4, 3),
    'font.size' : 10
    }

def testrdf(coordinates,boundary,rmax,dr):
    
    rvals = np.arange(0,rmax+dr,dr)
    
    density = len(coordinates)/np.product(boundary[:,1]-boundary[:,0])
    global dist
    tree = cKDTree(coordinates)
    dist,indices = tree.query(coordinates,k=len(coordinates),distance_upper_bound=rmax)
    
    mask = np.isfinite(dist) & (dist>0)
    dist = np.ma.masked_array(dist,mask)
    counts = np.apply_along_axis(lambda row: np.histogram(row.data[row.mask],bins=rvals)[0],1,dist)
    boundarycorr=_sphere_shell_vol_fraction(rvals, boundary-coordinates[:,:,np.newaxis])
    counts = np.sum(counts/boundarycorr,axis=0)

    
    print(counts)
    
    counts = counts / (4/3*np.pi * (rvals[1:]**3 - rvals[:-1]**3)) / (density*len(coordinates))
    
    return rvals,counts


#%% run iteration EXPERIMENTAL

plt.close('all')

#files = glob('*.dat')[::]
files = glob('**/*.dat',recursive=True)[::]

rmin = 1.8
rmax = 10
dr = 0.2
coordinateinterval = 1

rdfs = []
coords = []

print('loading and calculating g(r)')
for i,file in enumerate(files):
    print('\r{:} of {:}'.format(i+1,len(files)),end='',flush=True)
    testcoords,bb,_ = dumpreader_pandas(file)
    rvals,rdf = tp.pair_correlation_3d(testcoords,rmax,boundary=np.ravel(bb),dr=dr,handle_edge=True,max_rel_ndensity=50)
    rdfs.append(rdf)
    if i%coordinateinterval==0:
        coords.append(testcoords[['z','y','x']].to_numpy())
print()

rdf = np.mean(rdfs,axis=0)
rvals = np.arange(0,rmax+dr,dr)
rdf = rdf[rvals[:-1]>=rmin]
rvals = np.arange(rmin,rmax+dr,dr)
rcent = (rvals[1:] + rvals[:-1])/2
rdf[rdf<1e-20] = 1e-20
guess = rdf.copy()
guess = -np.log(guess)

print('starting iterations')
chi,pairpotential,paircorrelation,counters = run_iteration(
    coords[::],
    rdf,
    bb[::-1],
    #initial_guess=guess,
    rmin=rmin,
    rmax=rmax,
    dr=dr,
    max_iterations=50,
    convergence_tol=1e-6,
    n_ins=1000,
    grid=False,
    zero_clip=1e-20,
    periodic_boundary=False,
    )

cmap = get_cmap('jet')
colors = [cmap(i/(len(chi)-1))[:3]+(0.3,) for i in range(len(chi))]

with plt.rc_context(params):
    fig = plt.figure()
    plt.plot(rcent,rdf,label='distance histogram',color='k')
    [plt.plot(rcent,paircorrelation[i],label='particle insertion {:}'.format(i),color=colors[i]) for i in range(len(chi))]
    plt.xlim(rmin,rmax)
    plt.xlabel('r ($\mathrm{\mu m}$)')
    plt.ylabel('g(r)')
    plt.tight_layout()

with plt.rc_context(params):
    fig = plt.figure()
    [plt.plot(rcent,pairpotential[i],label='particle insertion {:}'.format(i),color=colors[i]) for i in range(len(chi))]
    plt.xlim(rmin,rmax)
    plt.tight_layout()

with plt.rc_context(params):
    fig=plt.figure()
    fig.set_size_inches(3,2.25)
    plt.plot(range(1,len(chi)+1),chi)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('iteration step')
    plt.ylabel('χ²')
    plt.tight_layout()
    plt.savefig('particle_insertion_chi.png',dpi=600,transparent=True)
    plt.savefig('particle_insertion_chi.pdf',transparent=True)

with plt.rc_context(params):
    plt.figure()
    plt.axhline(1,color='k',linewidth=0.75)
    plt.plot(rcent,rdf,'-r')
    plt.xlabel('$r$ ($\mathrm{\mu m}$)')
    plt.ylabel('$g(r)$')
    plt.xlim(rmin,rmax)
    plt.tight_layout()
    plt.savefig('histogram_binning_rdf.png',dpi=600,transparent=True)
    plt.savefig('histogram_binning_rdf.pdf',transparent=True)

with plt.rc_context(params):
    plt.figure()
    plt.axhline(1,color='k',linewidth=0.75)
    plt.plot(rcent,rdf,'-r',label='distance histogram')
    plt.plot(rcent,paircorrelation[-1],'o',color='k',markerfacecolor='none',label='final iteration')
    plt.xlabel('$r$ ($\mathrm{\mu m}$)')
    plt.ylabel('$g(r)$')
    plt.xlim(rmin,rmax)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('particle_insertion_rdf.png',dpi=600,transparent=True)
    plt.savefig('particle_insertion_rdf.pdf',transparent=True)

with plt.rc_context(params):
    plt.figure()
    plt.axhline(0,color='k',linewidth=0.75)
    plt.plot(rcent,pairpotential[-1],'-o',color='k',markeredgecolor='k',markerfacecolor='none')
    plt.xlabel('$r$ ($\mathrm{\mu m}$)')
    plt.ylabel('$u(r)$ $(k_B T)$')
    plt.xlim([rmin,rmax])
    plt.ylim([-0.5,10])
    plt.tight_layout()
    plt.savefig('particle_insertion_pairpotential.png',dpi=600,transparent=True)
    plt.savefig('particle_insertion_pairpotential.pdf',transparent=True)

sys.exit()

#%% run iteration PERIODIC

plt.close('all')

files = glob('*.dat')[::]
#files = glob('**/*.dat',recursive=True)[::20]

rmin = 0
rmax = 15
dr = 0.2
coordinateinterval = 1

rdfs = []
counts = []
coords = []

print('loading and calculating g(r)')
for i,file in enumerate(files):
    print('\r{:} of {:}'.format(i,len(files)),end='',flush=True)
    testcoords,bb,_ = dumpreader_pandas(file)
    rvals,rdf,count = pair_correlation_3d_periodic(testcoords,rmax,bb,rmin=rmin,dr=dr,handle_edge=True)
    rdfs.append(rdf)
    counts.append(count)
    if i%coordinateinterval==0:
        coords.append(testcoords[['z','y','x']].to_numpy())
print()

raise ValueError
rdf = np.mean(rdfs,axis=0)
count = np.sum(counts,axis=0)
#rdf[count!=0] /= count[count!=0]

rdf = rdf[rvals[:-1]>=rmin]
rcent = (rvals[1:] + rvals[:-1])/2
rdf[rdf<1e-20] = 1e-20
guess = rdf.copy()
guess = -np.log(guess)

print('starting iterations')
chi,pairpotential,paircorrelation,counters = run_iteration(
    coords[::],
    rdf,
    bb[::-1],
    #initial_guess=guess,
    rmin=rmin,
    rmax=rmax,
    dr=dr,
    max_iterations=20,
    convergence_tol=1e-5,
    n_ins=200,
    grid=False,
    zero_clip=1e-20,
    periodic_boundary=True
    )

cmap = get_cmap('jet')
colors = [cmap(i/(len(chi)-1))[:3]+(0.3,) for i in range(len(chi))]

plt.figure()
plt.plot(rcent,rdf,label='distance histogram',color='k')
[plt.plot(rcent,paircorrelation[i],label='particle insertion {:}'.format(i),color=colors[i]) for i in range(len(chi))]
plt.xlabel('r ($\mathrm{\mu m}$')
plt.ylabel('g(r)')


plt.figure('potential')
[plt.plot(rcent,pairpotential[i],label='particle insertion {:}'.format(i),color=colors[i]) for i in range(len(chi))]
plt.ylim([-0.1,8])

with plt.rc_context(params):
    fig=plt.figure()
    fig.set_size_inches(3,2.25)
    plt.plot(range(1,len(chi)+1),chi)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('iteration step')
    plt.ylabel('χ²')
    plt.tight_layout()
    plt.savefig('YK_particle_insertion_chi.png',dpi=600,transparent=True)
    plt.savefig('YK_particle_insertion_chi.pdf',transparent=True)

with plt.rc_context(params):
    plt.figure()
    plt.axhline(1,color='k',linewidth=0.75)
    plt.plot(rcent,rdf,'-r')
    plt.xlabel('$r/\sigma$')
    plt.ylabel('$g(r)$')
    plt.xlim(0,rmax)
    plt.tight_layout()
    plt.savefig('histogram_binning_rdf.png',dpi=600,transparent=True)
    plt.savefig('histogram_binning_rdf.pdf',transparent=True)

with plt.rc_context(params):
    plt.figure()
    plt.axhline(1,color='k',linewidth=0.75)
    plt.plot(rcent,rdf,'-r',label='distance histogram')
    plt.plot(rcent,paircorrelation[-1],'o',color='k',markerfacecolor='none',markersize=5,label='final iteration')
    plt.xlabel('$r/\sigma$')
    plt.ylabel('$g(r)$')
    plt.xlim(0,rmax)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('YK_particle_insertion_rdf.png',dpi=600,transparent=True)
    plt.savefig('YK_particle_insertion_rdf.pdf',transparent=True)


x = np.linspace(0.1,rmax,200)
y = [u_yk(i,eps=100,debye=2,cutoff=12) for i in x]

with plt.rc_context(params):
    plt.figure()
    plt.axhline(0,color='k',linewidth=0.75)
    plt.plot(x,y,'-r')
    plt.xlabel('$r/\sigma$')
    plt.ylabel('pairwise potential $(k_B T)$')
    plt.xlim([0,rmax])
    plt.ylim([-0.5,8])
    plt.tight_layout()
    plt.savefig('YK_input_pairpotential.png',dpi=600,transparent=True)
    plt.savefig('YK_input_pairpotential.pdf',transparent=True)

with plt.rc_context(params):
    plt.figure()
    plt.axhline(0,color='k',linewidth=0.75)
    plt.plot(x,y,'-r',label='simulation input')
    plt.plot(rcent,pairpotential[-1],'o',color='k',markerfacecolor='none',markersize=5,label='final iteration')
    plt.xlabel('$r/\sigma$')
    plt.ylabel('$u(r)/ k_B T$')
    plt.xlim([0,rmax])
    plt.ylim([-0.5,8])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('YK_particle_insertion_pairpotential.png',dpi=600,transparent=True)
    plt.savefig('YK_particle_insertion_pairpotential.pdf',transparent=True)

sys.exit()


#%% run iteration NONPERIODIC

plt.close('all')

files = glob('*.dat')[::]
rmin = 6
rmax = 16
dr = 0.5

rdfs = []
coords = []

print('loading and calculating g(r)')
for i,file in enumerate(files):
    print('\r{:} of {:}'.format(i,len(files)),end='',flush=True)
    testcoords,bb,_ = dumpreader_pandas(file)
    rvals,rdf = tp.pair_correlation_3d(testcoords,rmax,boundary=np.ravel(bb),dr=dr,handle_edge=True,max_rel_ndensity=50)
    rdfs.append(rdf)
    if i%1==0:
        coords.append(testcoords[['z','y','x']].to_numpy())
print()

rdf = np.mean(rdfs,axis=0)
rvals = np.arange(0,rmax+dr,dr)
rdf = rdf[rvals[:-1]>=rmin]
rvals = np.arange(rmin,rmax+dr,dr)
rcent = (rvals[1:] + rvals[:-1])/2
rdf[rdf<1e-20] = 1e-20
guess = rdf.copy()
guess = -np.log(guess)

print('starting iterations')
chi,pairpotential,paircorrelation,counters = run_iteration(
    coords[::],
    rdf,
    bb[::-1],
    #initial_guess=guess,
    rmin=rmin,
    rmax=rmax,
    dr=dr,
    max_iterations=20,
    convergence_tol=1e-5,
    n_ins=100,
    grid=False,
    zero_clip=1e-20
    )

cmap = get_cmap('jet')
colors = [cmap(i/(len(chi)-1))[:3]+(0.3,) for i in range(len(chi))]

plt.figure()
plt.plot(rcent,rdf,label='distance histogram',color='k')
[plt.plot(rcent,paircorrelation[i],label='particle insertion {:}'.format(i),color=colors[i]) for i in range(len(chi))]
plt.xlabel('r ($\mathrm{\mu m}$')
plt.ylabel('g(r)')


plt.figure('potential')
[plt.plot(rcent,pairpotential[i],label='particle insertion {:}'.format(i),color=colors[i]) for i in range(len(chi))]
plt.ylim([-0.1,8])

with plt.rc_context(params):
    fig=plt.figure()
    fig.set_size_inches(3,2.25)
    plt.plot(range(1,len(chi)+1),chi)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('iteration step')
    plt.ylabel('χ²')
    plt.tight_layout()
    plt.savefig('particle_insertion_chi.png',dpi=600,transparent=True)

with plt.rc_context(params):
    plt.figure()
    plt.axhline(1,color='k',linewidth=0.75)
    plt.plot(rcent,rdf,'-r')
    plt.xlabel('$r/\sigma$')
    plt.ylabel('$g(r)$')
    plt.xlim(0,rmax)
    plt.tight_layout()
    plt.savefig('histogram_binning_rdf.png',dpi=600,transparent=True)

with plt.rc_context(params):
    plt.figure()
    plt.axhline(1,color='k',linewidth=0.75)
    plt.plot(rcent,rdf,'-r',label='distance histogram')
    plt.plot(rcent,paircorrelation[-1],'o',color='k',markerfacecolor='none',markersize=5,label='final iteration')
    plt.xlabel('$r/\sigma$')
    plt.ylabel('$g(r)$')
    plt.xlim(0,rmax)
    plt.legend()
    plt.tight_layout()
    plt.savefig('particle_insertion_rdf.png',dpi=600,transparent=True)


x = np.linspace(0,rmax,200)
y = [u_yk(i,eps=5,debye=1,cutoff=5) for i in x]

with plt.rc_context(params):
    plt.figure()
    plt.axhline(0,color='k',linewidth=0.75)
    plt.plot(x,y,'-r',label='simulation input')
    plt.plot(rcent,pairpotential[-1],'o',color='k',markerfacecolor='none',markersize=5,label='final iteration')
    plt.xlabel('$r/\sigma$')
    plt.ylabel('$u(r)/ k_B T$')
    plt.xlim([0,rmax])
    plt.ylim([-1.5,8])
    plt.legend()
    plt.tight_layout()
    plt.savefig('particle_insertion_pairpotential.png',dpi=600,transparent=True)

sys.exit()

#%% plot 3d scatter
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
cmap = get_cmap('jet')
colors = [cmap(i/(len(coords)-1))[:3]+(0.3,) for i in range(len(coords))]

reduced_boundary = np.array(bb[::-1])
reduced_boundary[:,0] += rmax
reduced_boundary[:,1] -= rmax

fig = plt.figure('full stack')
ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect('equal')
[ax.scatter(p[:,2],p[:,1],p[:,0],s=3,color=c) for p,c in zip(coords,colors)]
set_axes_equal(ax)

fig = plt.figure('within reduced boundary')
ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect('equal')
part = [p[(
    (p[:,0]>=reduced_boundary[0,0]) & (p[:,0]<reduced_boundary[0,1]) &
    (p[:,1]>=reduced_boundary[1,0]) & (p[:,1]<reduced_boundary[1,1]) &
    (p[:,2]>=reduced_boundary[2,0]) & (p[:,2]<reduced_boundary[2,1])
    )] for p in coords]
[ax.scatter(p[:,2],p[:,1],p[:,0],color=c) for p,c in zip(part,colors)]
set_axes_equal(ax)

sys.exit()


#%% fit potential

from scipy.optimize import curve_fit

diam = 1.8
fit_r_min = 1


def fitfun(x,eps=1,debye=1):
    return eps * np.exp(-(x-diam)/debye)/(x/diam)

fitpotential = pairpotential[-1][pairpotential[-1]<10]
fitr = rcent[pairpotential[-1]<10]
fitpotential = fitpotential[fitr>=fit_r_min]
fitr = fitr[fitr>=fit_r_min]

fit = curve_fit(
    fitfun,
    fitr,
    fitpotential,
    p0=[500,0.1],
    bounds=([0,0],[np.inf,np.inf]),
    )[0]

x = np.linspace(0.1,rmax,200)
y = [fitfun(i,eps=fit[0],debye=fit[1]) for i in x]

with plt.rc_context(params):
    plt.figure()
    plt.axhline(0,color='k',linewidth=0.75)
    plt.plot(x,y,'-r',label='fit')
    plt.plot(rcent,pairpotential[-1],'o',color='k',markerfacecolor='none',markersize=5,label='data')
    plt.xlabel('$r$ ($\mathrm{\mu m}$)')
    plt.ylabel('$u(r)/k_BT$')
    #plt.yscale('log')
    plt.xlim([0,rmax])
    plt.ylim([-0.5,10])
    plt.legend()
    plt.text(
        0.95,
        0.6,
        '$\epsilon={:.3g}$'.format(fit[0])+
        ' $\mathrm{k_BT}$\n$\kappa^{-1}='+
        '{:.3g}$'.format(fit[1])+
        ' $\mathrm{\mu m}$',
        ha='right',
        va='center',
        transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig('particle_insertion_pairpotential_fit.png',dpi=600,transparent=True)
    plt.savefig('particle_insertion_pairpotential_fit.pdf',transparent=True)