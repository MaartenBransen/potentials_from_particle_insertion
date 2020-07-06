#imports

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from glob import glob
from potentials_from_particle_insertion import run_iteration,rdf_dist_hist_3d


#%% data loader definition
def dumpreader_pandas(filename,scaled_box=False,periodic_boundary=False):
    """
    reads in a list of coordinate dump files
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

#%% run
files = glob('*.dat')[::]

#settings
rmin = 1
rmax = 5
dr = 0.2

#load data from files
coords = []
for file in files:
    coord,boundary,_ = dumpreader_pandas(file)
    coords.append(coord[['z','y','x']].to_numpy())
boundary = boundary[::-1]#reverse to get z,y,x

#calculate histogram binning g(r)
binedges,binvals = rdf_dist_hist_3d(
    coords,
    rmin=rmin,
    rmax=rmax,
    dr=dr,
    boundary=boundary,
    periodic_boundary=True
)

bincent = (binedges[1:]+binedges[:-1])/2

#plot rdf
plt.figure()
plt.plot(bincent,binvals,'-r',label='dist hist')
plt.xlabel('r ($\mathrm{\mu m}$)')
plt.ylabel('g(r)')

#run iterations
error,potential,rdf,counts = run_iteration(
    coords,
    binvals,
    boundary,
    rmin=rmin,
    rmax=rmax,
    dr=dr,
    max_iterations=10,
    convergence_tol=1e-5,
    n_ins=500,
    regulate=False,
    periodic_boundary=True
)

#add plot of final iteration
plt.plot(bincent,rdf[-1],'ok',markerfacecolor='none',label='insertion')
plt.legend()

#plot pairpotential from final iteration
plt.figure()
plt.plot(bincent,potential[-1],'-k',label='dist hist')
plt.xlabel('r ($\mathrm{\mu m}$)')
plt.ylabel('u(r) (k$_\mathrm{B}$T)')

#plot mean squared error
plt.figure()
plt.plot(range(1,len(error)+1),error)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('iteration')
plt.ylabel('$\chi^2$')
