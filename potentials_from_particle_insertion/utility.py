import numpy as np

def save_TPI_results(rmin,rmax,dr,bincentres,dhrdf,rdfs,potentials,counts,
                     errors,filename='particle_insertion_result.txt'):
    """
    utility function for saving results from TPI as txt file

    Parameters
    ----------
    rmin : float
        lower bound on interparticle distance
    rmax : float
        upper bound on interparticle distance
    dr : float
        bin width of interparticle distance
    bincentres : list of float
        centre position of each interparticle distance bin
    dhrdf : list of float
        g(r) values from DH calculation for each bin
    rdfs : list of list of float
        g(r) values from TPI for each bin for each iteration
    potentials : list of list of float
        u(r) values for each bin for each iteration
    counts : list of list of float
        bin counts for each bin for each iteration
    errors : list of float
        chi squared error for each iteration
    filename : str, optional
        filename to use. The default is 'particle_insertion_result.txt'.
    """
    #open file, overwrite if already exists
    with open(filename,'w+') as file:
        
        #input params
        file.write('rmin:\t{:}\n'.format(rmin))
        file.write('rmax:\t{:}\n'.format(rmax))
        file.write('dr: \t{:}\n'.format(dr))
        file.write('N_iter:\t{:}\n'.format(len(potentials)))
        file.write('N_r:\t{:}\n'.format(len(bincentres)))
        file.write('\n')
        
        #write DH rdf
        file.write('#          r\t     g_DH(r)\n')
        for r,g in zip(bincentres,dhrdf):
            file.write('{: .5e}\t{: .5e}\n'.format(r,g))
        
        #write TPI table
        for i,(rdf,pot,count,err) in enumerate(zip(rdfs,potentials,counts,errors)):
            file.write('\niteration:\t{}\n'.format(i))
            file.write('chi_squared:\t{:.5e}\n'.format(err))
            file.write('   g_TPI(r)\t        u(r)\t    counts\n')
            for g,u,c in zip(rdf,pot,count):
                file.write('{: .5e}\t{: .5e}\t{: 6d}\n'.format(g,u,int(c)))
        

def load_TPI_results(filename):
    """
    loads text file containing TPI results as written by `save_tpi_results()`

    Parameters
    ----------
    filename : str
        name of the TPI result file to load

    Returns
    -------
    rmin : float
        lower bound on interparticle distance
    rmax : float
        upper bound on interparticle distance
    dr : float
        bin width of interparticle distance
    bincentres : list of float
        centre position of each interparticle distance bin
    dhrdf : list of float
        g(r) values from DH calculation for each bin
    rdfs : list of list of float
        g(r) values from TPI for each bin for each iteration
    potentials : list of list of float
        u(r) values for each bin for each iteration
    counts : list of list of float
        bin counts for each bin for each iteration
    errors : list of float
        chi squared error for each iteration
    """
    
    #open file and read data
    with open(filename,'r') as file:
        filedata = file.readlines()
    
    #load header
    rmin = float(filedata[0].split()[1])
    rmax = float(filedata[1].split()[1])
    dr   = float(filedata[2].split()[1])
    Ni   = int(filedata[3].split()[1])
    Nr   = int(filedata[4].split()[1])
    
    bincentres,dhrdf = [],[]
    for line in filedata[7:7+Nr]:
        r,g = line.split()
        bincentres.append(float(r))
        dhrdf.append(float(g))
    bincentres = np.array(bincentres)
    dhrdf = np.array(dhrdf)
    
    
    #init lists
    rdfs = []
    potentials = []
    counts = []
    errors = []
    
    #load table
    for i in range(Ni):
        start = 8+Nr + (Nr+4)*i
        errors.append(float(filedata[start+1].split()[1]))
        rdf = []
        potential = []
        count = []
        for line in filedata[start+3:start+3+Nr]:
            g,u,c = line.split()
            rdf.append(float(g))
            potential.append(float(u))
            count.append(int(c))
        rdfs.append(np.array(rdf))
        potentials.append(np.array(potential))
        counts.append(np.array(count))
        
    return rmin,rmax,dr,bincentres,dhrdf,rdfs,potentials,counts,errors