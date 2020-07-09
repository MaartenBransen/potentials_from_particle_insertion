#imports

from matplotlib import pyplot as plt
import numpy as np
from potentials_from_particle_insertion import run_iteration,rdf_dist_hist_3d
from potentials_from_particle_insertion.generate_coordinates import _rand_coord_in_box

#%% input parameters

#data generation
boxsize = 10
n = 1000 #number of particles
m = 100  #number of datasets

#g(r)
rmin = 0
rmax = 5
dr = 0.1

#%% create fake data
boundary = np.array([[0,boxsize]]*3)
coords = [_rand_coord_in_box(boundary,n=n) for _ in range(m)]


#%% calculate distance histogram g(r)

#calculate histogram binning g(r)
binedges,binvals = rdf_dist_hist_3d(
    coords,
    rmin=rmin,
    rmax=rmax,
    dr=dr,
    boundary=boundary,
    periodic_boundary=False
)

bincent = (binedges[1:]+binedges[:-1])/2

#plot rdf
plt.figure()
plt.plot(bincent,binvals,'-r',label='dist hist')
plt.xlabel('r ($\mathrm{\mu m}$)')
plt.ylabel('g(r)')
plt.xlim(rmin,rmax)


#%% run iterations
error,potential,rdf,counts = run_iteration(
    coords,
    binvals,
    boundary,
    rmin=rmin,
    rmax=rmax,
    dr=dr,
    max_iterations=10,
    convergence_tol=1e-6,
    n_ins=500,
    regulate=True
)


#%% plots
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

from matplotlib.cm import get_cmap
cmap = get_cmap('jet')
colors = [cmap(i/(len(error)-1))[:3]+(0.3,) for i in range(len(error))]

with plt.rc_context(params):
    fig = plt.figure()
    plt.plot(bincent,binvals,label='distance histogram',color='k')
    [plt.plot(bincent,rdf[i],label='particle insertion {:}'.format(i),color=colors[i]) for i in range(len(error))]
    plt.xlim(rmin,rmax)
    plt.xlabel('r ($\mathrm{\mu m}$)')
    plt.ylabel('g(r)')
    plt.tight_layout()

with plt.rc_context(params):
    fig = plt.figure()
    [plt.plot(bincent,potential[i],label='particle insertion {:}'.format(i),color=colors[i]) for i in range(len(error))]
    plt.xlim(rmin,rmax)
    plt.xlabel('r ($\mathrm{\mu m}$)')
    plt.ylabel('u(r) $(k_B T)$')
    plt.tight_layout()

with plt.rc_context(params):
    fig=plt.figure()
    plt.plot(range(1,len(error)+1),error)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('iteration step')
    plt.ylabel('χ²')
    plt.tight_layout()

with plt.rc_context(params):
    plt.figure()
    plt.axhline(1,color='k',linewidth=0.75)
    plt.plot(bincent,binvals,'-r',label='distance histogram')
    plt.plot(bincent,rdf[-1],'o',color='k',markerfacecolor='none',label='final iteration')
    plt.xlabel('$r$ ($\mathrm{\mu m}$)')
    plt.ylabel('$g(r)$')
    plt.xlim(rmin,rmax)
    plt.legend(frameon=False)
    plt.tight_layout()

with plt.rc_context(params):
    plt.figure()
    plt.axhline(0,color='k',linewidth=0.75)
    plt.plot(bincent,potential[-1],'-o',color='k',markeredgecolor='k',markerfacecolor='none')
    plt.xlabel('$r$ ($\mathrm{\mu m}$)')
    plt.ylabel('$u(r)$ $(k_B T)$')
    plt.xlim([rmin,rmax])
    plt.tight_layout()
