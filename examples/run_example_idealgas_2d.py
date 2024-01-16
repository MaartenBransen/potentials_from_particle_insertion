"""
Copyright (c) 2024 Maarten Bransen
"""


#imports

from matplotlib import pyplot as plt
import numpy as np
from potentials_from_particle_insertion import run_iteration,rdf_dist_hist_2d
from potentials_from_particle_insertion.generate_coordinates import _rand_coord_in_box


#%% input parameters

#data generation
boxsize = 10
n = 1000 #number of particles
m = 100  #number of datasets

#g(r)
rmin = 0
rmax = 6
dr = 0.1

# create fake data
boundary = np.array([[0,boxsize]]*2)
coords = [_rand_coord_in_box(boundary,n=n) for _ in range(m)]


#%% calculate distance histogram g(r)

#calculate histogram binning g(r)
binedges,binvals = rdf_dist_hist_2d(
    coords,
    rmin=rmin,
    rmax=rmax,
    dr=dr,
    handle_edge='periodic rectangle',
    boundary=boundary,
)

bincent = (binedges[1:]+binedges[:-1])/2

#plot rdf
plt.figure('distance histogram g(r)')
plt.plot(bincent,binvals,'-r',label='dist hist')
plt.xlabel('r')
plt.ylabel('g(r)')
plt.xlim(rmin,rmax)
plt.ylim(0,1.1)
plt.show()


#%% run iterations
error,potential,rdf,counts = run_iteration(
    coords,
    binvals,
    rmin=rmin,
    rmax=rmax,
    dr=dr,
    handle_edge='periodic rectangle',
    boundary=boundary,
    max_iterations=5,
    convergence_tol=1e-7,
    n_ins=500,
    regulate=True,
)


#%% plots
from matplotlib.cm import get_cmap
cmap = get_cmap('jet')
colors = [cmap(i/(len(error)-1))[:3]+(0.3,) for i in range(len(error))]

#plot all iterations on top of eachother
fig = plt.figure('g(r) evolution')
plt.plot(bincent,binvals,label='distance histogram',color='k')
[plt.plot(bincent,rdf[i],label='particle insertion {:}'.format(i),color=colors[i]) for i in range(len(error))]
plt.xlim(rmin,rmax)
plt.ylim([0,1.1])
plt.xlabel('$r$')
plt.ylabel('$g(r)$')
plt.tight_layout()
plt.show()

#plot u(r) for all iterations on top of eachother
fig = plt.figure('u(r) evolution')
[plt.plot(bincent,potential[i],label='particle insertion {:}'.format(i),color=colors[i]) for i in range(len(error))]
plt.xlim(rmin,rmax)
plt.xlabel('$r$')
plt.ylabel('$u(r)$')
plt.tight_layout()
plt.show()

#plot chi / error as function of iteration number
fig=plt.figure('error')
plt.plot(range(1,len(error)+1),error)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('iteration step')
plt.ylabel('χ²')
plt.tight_layout()
plt.show()

#plot distance histogram and last iteration together
plt.figure('final g(r)')
plt.axhline(1,color='k',linewidth=0.75)
plt.plot(bincent,binvals,'-r',label='distance histogram')
plt.plot(bincent,rdf[-1],'o',color='k',markerfacecolor='none',label='final iteration')
plt.xlabel('$r$')
plt.ylabel('$g(r)$')
plt.xlim(rmin,rmax)
plt.ylim([0,1.1])
plt.legend()
plt.tight_layout()
plt.show()

#plot u(r) of last iteration
plt.figure('final pair potential')
plt.axhline(0,color='k',linewidth=0.75)
plt.plot(bincent,potential[-1],'-o',color='k',markeredgecolor='k',markerfacecolor='none')
plt.xlabel('$r$')
plt.ylabel('$u(r)$')
plt.xlim([rmin,rmax])
plt.tight_layout()
plt.show()
