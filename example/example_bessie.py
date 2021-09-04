import os
import sys
#Add the path to the package.
path = os.path.abspath(os.getcwd())
package_path = os.path.join(path ,'../..')
sys.path.append(package_path) 

import numpy as np 
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import bessie 
import time
"""
cubic_spline_horn.py 

Example script of how to use bessie with non-trivial inputs

"""

########################
# Define the output.
# Create directory if it does not exist

outpath = os.path.join(path ,'data/')
tag = 'cubic_spline'
print('Saving data to path {}'.format(outpath))

if not os.path.exists(outpath):
    os.makedirs(outpath)

#########################
# Define the horn parameters 
 
r0 = 0.676
rend = 5.0
num_elems = 100
r_nodes = [r0, 1.5, 3.5, rend]
z_nodes = [0, 10, 20, 30]
rad_array, height_array, _ = bessie.cubic_spline_wg(num_elems, r_nodes, z_nodes, bc_type = 'natural', units = 'mm') #The outputs of this in in meters 

geometry = (height_array, rad_array)
horn = bessie.HornMM(geometry, units = 'm')
horn.export_geometry(units = 'mm', outdir = outpath,tag = tag) 
horn.plot_geometry(draw_frame = True, outdir = outpath)

#######################################
# Perform the calculation 
# Can be easily parallizable in this step as
# each freq is performed independently. 

startf = 140 #in GHz
endf = 175 #in GHz 
stepf = 5 #in GHz
num_modes = 10 #Number of radial models per type of mode 
freq = [] 

filename = lambda f, sparam: outpath + tag + '_s{}_f{}.npy'.format(f, sparam) 

#Loop over all frequencies and run the code 
freq_sweep = np.arange(startf, endf, stepf)
total_s21 = np.zeros((num_modes*2, len(freq_sweep)))
for i, f in enumerate(freq_sweep): 
    print('Computing f: {}'.format(f)) 
    ti = time.time()
    #Cascade the s-matrix
    smatrix = horn.computeTotalSMatrix(f, num_modes, outpath, tag = tag) 
    tf = time.time() 
    print('Done with s matrix: {} sec'.format(tf - ti))
    t1 = time.time()
    #Compute the far field beam pattern
    horn.computeFarFieldBeam(f, num_modes, outpath, tag = tag, deltaTheta = np.deg2rad(0.5), deltaPhi = np.deg2rad(2))
    #Create the summary pots 
    horn.summary_plots(f, outpath, tag = tag)
    
    t2 = time.time() 
    print('Frequency {} GHz, time: {} sec'.format(f, t2 - t1)) 

    print('Total time is {} sec'.format(t2 - ti))

    #Collect the modal structure 
    s21 = smatrix['s21'] 
    for j in np.arange(0, num_modes*2): 
        total_s21[j, i] = np.abs(s21[j,0])

################################
#Create the transmission plot 
################################
for j in np.arange(0, int(num_modes/2)): 
    mlabel = '1{}'.format(j + 1)
    label = r'TE$_{' + mlabel + '}$'
    plt.plot(freq_sweep, total_s21[j,:], label = label)

    mlabel = '1{}'.format(j + 1)
    label = r'TM$_{' + mlabel + '}$'
    plt.plot(freq_sweep, total_s21[j + num_modes,:], linestyle = '--', label = label)

plt.xlabel('Frequency (GHz)') 
plt.ylabel('Transmission |$s_{j1}$|')
plt.legend() 
plt.savefig(outpath + tag + '_s21.png')
plt.show() 
