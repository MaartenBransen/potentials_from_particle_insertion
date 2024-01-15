"""
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

#imports
from matplotlib import pyplot as plt
import numpy as np
from potentials_from_particle_insertion.multicomponent import run_iteration,rdf_dist_hist_3d
from potentials_from_particle_insertion.generate_coordinates import _rand_coord_in_box, _rand_coord_at_dist #only for generating fake data

#%% input parameters

#data generation
boxsize = 10
n = 200 #number of particles / component
m = 100  #number of datasets

#g(r)
rmin = 0
rmax = 5
dr = 0.1

# create fake data
boundary = np.array([[0,boxsize]]*3)
coords = []
for _ in range(m):
    #randomly place c0, i.e. g(r)=1
    c0 = _rand_coord_in_box(boundary,n=n)
    #place c1 particles at least distance 1 from c0
    c1 = _rand_coord_at_dist(boundary,c0,1,n=n)
    coords.append([c0,c1])


#simple fit function
def fitfun(r,a,b):
    res = a*r+b
    res[r>3] = 0
    res[r<0] = np.inf
    return res

#%% calculate distance histogram g(r)

#calculate histogram binning g(r)
binedges,binvals,combinations = rdf_dist_hist_3d(
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
[plt.plot(bincent,bv,'-',label=com) for bv,com in zip(binvals,combinations)]
plt.xlabel('$r$')
plt.ylabel('$g(r)$')
plt.xlim(rmin,rmax)
plt.ylim(0,1.1)
plt.legend(title='combination')
plt.show()

#%% run iterations
error,potential,rdf,counts,combinations = run_iteration(
    coords,
    binvals,
    rmin=rmin,
    rmax=rmax,
    dr=dr,
    max_iterations=10,
    convergence_tol=1e-7,
    n_ins=200,
    boundary=boundary,
    handle_edge='cuboid',
    regulate=True,
)

#%% plots
from matplotlib.cm import get_cmap
cmap = get_cmap('jet')
colors = [cmap(i/(len(error)-1))[:3]+(0.5,) for i in range(len(error))]

rdf_fig,rdf_ax = plt.subplots(len(combinations),sharex=True,figsize=(6,3*len(combinations)))
pot_fig,pot_ax = plt.subplots(len(combinations),sharex=True,figsize=(6,3*len(combinations)))

for j,com in enumerate(combinations):
    #plot all iterations on top of eachother
    [rdf_ax[j].plot(bincent,rdf[i][j],label=i,color=colors[i]) for i in range(len(error))]
    rdf_ax[j].plot(bincent,binvals[j],':',label='DH',color='k')
    rdf_ax[j].set_title(com)
    rdf_ax[j].set_ylabel('$g(r)$')
    rdf_ax[j].set_xlim(rmin,rmax)
    #rdf_ax[j].legend()
    
    #plot u(r) for all iterations on top of eachother
    [pot_ax[j].plot(bincent,potential[i][j],label=i,color=colors[i]) for i in range(len(error))]
    pot_ax[j].set_title(com)
    pot_ax[j].set_ylabel('$u(r)$ $(k_B T)$')
    pot_ax[j].set_xlim(rmin,rmax)
    #pot_ax[j].legend()

rdf_ax[0].legend(title='iteration',ncol=2)
rdf_ax[0].set_xlabel('$r$')
rdf_fig.tight_layout()
rdf_fig.show()

pot_ax[0].legend(title='iteration',ncol=2)
pot_ax[0].set_xlabel('$r$')
pot_fig.tight_layout()
pot_fig.show()
    

#plot distance histogram and last iteration together
plt.figure()
plt.title('last iteration')
plt.axhline(1,color='k',linewidth=0.75)
for j,com in enumerate(combinations):
    line, = plt.plot(bincent,binvals[j],'-',label=f'DH {com}')
    plt.plot(bincent,rdf[-1][j],'o',c=line.get_color(),mfc='none',label=f'TPI {com}')
plt.xlabel('$r$')
plt.ylabel('$g(r)$')
plt.xlim(rmin,rmax)
plt.legend()
plt.tight_layout()
plt.show()

#plot u(r) of last iteration
plt.figure()
plt.title('last iteration')
plt.axhline(0,color='k',linewidth=0.75)
for j,com in enumerate(combinations):
    plt.plot(bincent,potential[-1][j],'-o',mfc='none',label=com)
plt.xlabel('$r$')
plt.ylabel('$u(r)$ $(k_B T)$')
plt.xlim([rmin,rmax])
plt.legend()
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
    
plt.show()
