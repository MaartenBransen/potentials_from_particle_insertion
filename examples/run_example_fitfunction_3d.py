#imports

from matplotlib import pyplot as plt
import numpy as np
from potentials_from_particle_insertion import run_iterator_fitfunction,rdf_dist_hist_3d
from potentials_from_particle_insertion.generate_coordinates import _rand_coord_in_box #only for generating fake data
plt.close('all')

#%% input parameters

#data generation
boxsize = 10
n = 1000 #number of particles
m = 100  #number of datasets

#g(r)
rmin = 0
rmax = 5
dr = 0.1

# create fake data
boundary = np.array([[0,boxsize]]*3)
coords = [_rand_coord_in_box(boundary,n=n) for _ in range(m)]

#simple fit function
def fitfun(r,a,b):
    res = a*r+b
    res[r>3] = 0
    res[r<0] = np.inf
    return res

#%% calculate distance histogram g(r)

#calculate histogram binning g(r)
binedges,binvals = rdf_dist_hist_3d(
    coords,
    rmin=rmin,
    rmax=rmax,
    dr=dr,
    boundary=boundary,
    handle_edge='cuboid',
)

bincent = (binedges[1:]+binedges[:-1])/2

#plot rdf
plt.figure()
plt.plot(bincent,binvals,'-r',label='dist hist')
plt.xlabel('r ($\mathrm{\mu m}$)')
plt.ylabel('g(r)')
plt.xlim(rmin,rmax)
plt.ylim(0,1.1)
plt.show()

#%% run iterations
error,potential,fitparams,rdf,counts = run_iterator_fitfunction(
    coords,
    binvals,
    fitfun,
    rmin=rmin,
    rmax=rmax,
    dr=dr,
    initial_guess=[-0.1,0.3],
    fit_bounds = [[-np.inf,0],[0,np.inf]],
    max_iterations=10,
    convergence_tol=1e-6,
    n_ins=1000,
    boundary=boundary,
    handle_edge='cuboid',
)


#%% plots
from matplotlib.cm import get_cmap
cmap = get_cmap('jet')
colors = [cmap(i/(len(error)-1))[:3]+(0.3,) for i in range(len(error))]

#plot all iterations on top of eachother
fig = plt.figure()
plt.plot(bincent,binvals,label='distance histogram',color='k')
[plt.plot(bincent,rdf[i],label='particle insertion {:}'.format(i),color=colors[i]) for i in range(len(error))]
plt.xlim(rmin,rmax)
plt.xlabel('r ($\mathrm{\mu m}$)')
plt.ylabel('g(r)')
plt.tight_layout()

#plot u(r) for all iterations on top of eachother
plotx = np.linspace(rmin,rmax,200)
fig = plt.figure()
[plt.plot(plotx,potential[i](plotx),label='particle insertion {:}'.format(i),color=colors[i]) for i in range(len(error))]
plt.xlim(rmin,rmax)
plt.xlabel('r ($\mathrm{\mu m}$)')
plt.ylabel('u(r) $(k_B T)$')
plt.tight_layout()
plt.show()

#plot chi / error as function of iteration number
fig=plt.figure()
plt.plot(range(1,len(error)+1),error)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('function evaluation')
plt.ylabel('χ²')
plt.tight_layout()

#plot distance histogram and last iteration together
plt.figure()
plt.axhline(1,color='k',linewidth=0.75)
plt.plot(bincent,binvals,'-r',label='distance histogram')
plt.plot(bincent,rdf[-1],'o',color='k',markerfacecolor='none',label='final iteration')
plt.xlabel('$r$ ($\mathrm{\mu m}$)')
plt.ylabel('$g(r)$')
plt.xlim(rmin,rmax)
plt.legend(frameon=False)
plt.tight_layout()

#plot u(r) of last iteration
plt.figure()
plt.axhline(0,color='k',linewidth=0.75)
plt.plot(plotx,potential[-1](plotx),'-o',color='k',markeredgecolor='k',markerfacecolor='none')
plt.xlabel('$r$ ($\mathrm{\mu m}$)')
plt.ylabel('$u(r)$ $(k_B T)$')
plt.xlim([rmin,rmax])
plt.tight_layout()
    
plt.show()
