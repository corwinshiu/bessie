import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
#from matplotlib.widgets import TextBox
#import CircWGNorm as circ
from bessie import CircWGNorm as circ
from scipy.special import jv
from scipy.special import jvp
from scipy import special
from scipy.integrate import dblquad
import mpl_toolkits.mplot3d.axes3d as axes3d
import time 

class HornMM:
    def __init__(self, geometry, conductivity = np.inf, units = 'm'):

        """
        HornMM class

        Arguments
        ---------
        geometry : tuple
            A tuple of the form (zval, rval)
            that describes an axially symmetric waveguide horn as
            a collection of circular waveguides with radius rval
            and height zval
        conductivity : float. 
            Conductivity of the material in SI units (siemens/m). 
            default is perfect electrical conductor.  
        units : string (default: 'm' = meters)
        """
           
        zlist = np.array(geometry[0])
        rlist = np.array(geometry[1])
        if units == 'm':
            pass
        elif units == 'cm':
            conductivity *= 1.0E2
            zlist = zlist*1.0E-2 
            rlist = rlist*1.0E-2 
        elif units == 'mm':
            conductivity *= 1.0E3
            zlist = zlist*1.0E-3 
            rlist = rlist*1.0E-3 
        else: 
            print('Units not recoganized. Please input units in meters') 
            exit() 

        self.conductivity = conductivity


        #If zlist and rlist are the same sized, add an extra term to the r list to terminate the horn with a circular waveguide. 
        if len(zlist) == len(rlist):
            rlist = np.append(rlist, rlist[-1]) 
            self.geometry = (zlist, rlist)
        else: 
            self.geometry = geometry
    def export_geometry(self, units = 'm', outdir = '', tag = ''):
        """
        Export the geomtery of the horn in a csv format 
        
        Arguments
        ---------
        units : str (default meters). 
        outdir : str (default ''). 
        tag : str (default ''). 
            A unique label for recordkeeping. 
        """
        zlist = np.array(self.geometry[0])
        rlist = np.array(self.geometry[1])
        rlist = rlist[:-1]

        if units == 'm':
            pass
        elif units == 'cm':
            zlist *= 1.e2
            rlist *= 1.e2
        elif units == 'mm':
            rlist *= 1.e3
            zlist *= 1.e3
            
        geo_data = np.array([zlist, rlist]).T


        filename = outdir + tag + '_geometry.csv'
        np.savetxt(filename, geo_data, delimiter = ',')
        
        
    def plot_geometry(self, draw_frame = False, outdir = ''):
        """ 
        Plot the geometry of the horn.

        Arguments
        ---------
        draw_frame : boolean
            Include a border illustrating a cross section of the horn
        outpath : str (default '')
            Location of the output file.
        """
        #Create the points 
        total_segs = len(self.geometry[0])
        zvals = self.geometry[0]
        rvals = self.geometry[1]

        z0 = 0
        rvals_plot = [] 
        zvals_plot = []
        for index in np.arange(0, total_segs): 
            rvals_plot.append(rvals[index]) 
            rvals_plot.append(rvals[index])

            zvals_plot.append(z0)
            z0 = z0 + zvals[index] 
            zvals_plot.append(z0) 

        rvals_plot2 = [-1*r for r in rvals_plot]
        rvals_plot = np.array(rvals_plot) 
        rvals_plot2 = np.array(rvals_plot2) 
        zvals_plot = np.array(zvals_plot)
        self.geometry_plot = (zvals_plot, rvals_plot)
        fig = plt.figure() 
        ax = fig.add_subplot(111) 

        if draw_frame: 
            border = 1.
            top = np.max(rvals_plot)*1.0E3  + border
            bot = np.min(rvals_plot2)*1.0E3 - border
            right = np.max(zvals_plot)*1.0E3
            ax.plot([0,0],[bot, rvals_plot2[0]*1.0E3], color = 'k') 
            ax.plot([0,0], [top, rvals_plot[0]*1.0E3], color = 'k')
            ax.plot([0,right], [bot, bot], color = 'k') 
            ax.plot([0, right],[top, top], color = 'k') 
            ax.plot([right, right],[bot, rvals_plot2[-1]*1.0E3], color = 'k')
            ax.plot([right, right],[top, rvals_plot[-1]*1.0E3], color ='k')

        ax.plot(zvals_plot*1.0E3, rvals_plot*1.0E3, color = 'k')
        ax.plot(zvals_plot*1.0E3, rvals_plot2*1.0E3, color = 'k')
        ax.set_xlabel('Z-axis (mm)')
        ax.set_ylabel('R-axis (mm)')
        ax.set_aspect('equal')
        plt.savefig(outdir + 'HornGeometry.png')
        plt.show()
        plt.close()

        
    def cascadeTwoSMatrix(self, smatrix1, smatrix2, betaElls):
        """ 
        Cascade two S-matrixes. Each S-matrix is a block matrix 
        consisting of s11, s12, s21, s22 subblocks. 
        Each subblock is a 2n x 2n matrix consisting of TE+TM modes. 
        where n is the number of mode (TE1N,TE1N). 
        A phase delay of betaElls is included. 

        Returns
        -------
        smatrix: NumPy array (4n x 4n). n = number of modes.

        Arguments
        ---------
        smatrix1: NumPy array (4n x 4n). n = number of modes.
        smatrix2: NumPy array (4n x 4n).
        betaElls: phase delay by a transmission line. 
        """

        #SMatrix is a block matrix consisting of S11,S12,S21,S22 subblocks.
        # Each subblock is 2n x 2n accounting for TE and TM modes.
        num_modes = smatrix2.shape[0]/4
        n = int(2*num_modes) #Index associated with each subblock.


        s11 = smatrix1[:n, :n]
        s12 = smatrix1[:n,n:]
        s21 = smatrix1[n:, :n]
        s22 = smatrix1[n:, n:]
        
        s11p = smatrix2[:n, :n]
        s12p = smatrix2[:n, n:]
        s21p = smatrix2[n:, :n]
        s22p = smatrix2[n:,n:]
        
        ident = np.identity(n) 

        #Cascade the s-matrix
        s11c = s11 + s12@betaElls@np.linalg.inv(ident - s11p@betaElls@s22@betaElls)@s11p@betaElls@s21
        s12c = s12@betaElls@np.linalg.inv(ident - s11p@betaElls@s22@betaElls)@s12p
        s21c = s21p@betaElls@np.linalg.inv(ident - s22@betaElls@s11p@betaElls)@s21
        s22c = s22p + s21p@betaElls@np.linalg.inv(ident - s22@betaElls@s11p@betaElls)@s22@betaElls@s12p

        
        smatrix_full = np.block([[s11c, s12c],[s21c, s22c]])
        return smatrix_full
    
    #Freq should be in GHz
    def computeTotalSMatrix(self, freq, num_modes = 5, outdir = '', tag = 'horn'):
        """ 
        Compute the total SMatrix of the horn assembly. 

        Returns
        -------
        smatrix: dict
            keys consisting of s11, s12, s21, s22 blocks.

        Arguments
        ---------
        freq: float.  
            Frequency in units of GHz.
        num_modes: int. (default: 5)  
            Number of modes to include in the calculation. 
            I.E. 5 modes will include 5 TE modes, and 5 TM modes
        outdir: str (default: '') 
        tag: str(default: '') 
            Unique identifier for this horn. 
        """

        #Determine if a data file already exists - if so load from memory: 
        filename = outdir + tag + '_sparams_f{}ghz.npz'.format(int(freq))
        if os.path.exists(filename): 
            data = np.load(filename) 
            
            smatrix = {'s11': data['s11'],
                       's12': data['s12'],
                       's21': data['s21'],
                       's22': data['s22']}
            return smatrix




        zvals = self.geometry[0]
        rvals = self.geometry[1]

        inputWgRad = rvals[0]
        #Initalize the input mode (no reflection)
        smatrix_blockA = np.identity(num_modes*2, dtype = complex)
        smatrix_blockB = np.zeros((num_modes*2, num_modes*2), dtype = complex)
        smatrix_total = np.block([[smatrix_blockB, smatrix_blockA],[smatrix_blockA, smatrix_blockB]])

        #Cascade the s-matrixes recursively 
        for junc in np.arange(1, len(self.geometry[1])):

            #Get the smatrix at a waveguide junction
            smatrix_at_junc = circ.calculateSMatrix(freq*1.0E9, num_modes, rvals[junc - 1], rvals[junc])
            #Get the phase delay, accounting for any attenuation
            betaElls  = circ.getBetaElls(freq*1.0E9, rvals[junc - 1], zvals[junc - 1], num_modes, conductivity = self.conductivity)
            
            smatrix_total = self.cascadeTwoSMatrix(smatrix_total, smatrix_at_junc, betaElls)


        #We want to break this apart into s11_matrix, s21_matrix etc.
        s11_block = smatrix_total[0:2*num_modes, 0:2*num_modes]
        s21_block = smatrix_total[2*num_modes:, 0:2*num_modes]
        s12_block = smatrix_total[0:2*num_modes, 2*num_modes:]
        s22_block = smatrix_total[2*num_modes:, 2*num_modes:]
        
        ##########################################
        # Save the file 
        #########################################
        np.savez(filename, 
                 freq = freq,
                 num_modes = num_modes, 
                 geometry = {'rvals': rvals, 
                             'zvals': zvals}, 
                 s11 = s11_block,
                 s12 = s12_block,
                 s21 = s21_block,
                 s22 = s22_block)


        smatrix = {'s11': s11_block, 
                   's12': s12_block, 
                   's21': s21_block, 
                   's22': s22_block} 
        return smatrix 
    

    def computeApertureField(self, freq, num_modes = 5, outdir = '', tag = 'horn'):
        """ 
        Computes the electric and magnetic fields at the aperture
        and saves to disk. In summary we cascade the full s-matrix
        and then account for reflections at the horn interface to 
        first order terms. 

        Returns
        -------
        A tuple of (Efield, HField, s21) where each field is a vector 
        field that has the inputs (rho, phi), and outputs are 
        in (Ex, Ey) or (Hx, Hy) respectively. s21 is a 2n vector describing
        the modal content ordered TE and TM respectively. 
        
        Arguments
        ---------
        freq : float (in GHz). 
        num_modes: int (default: 5).
            Number of modes to propagate through the stacked waveguides. 
            Note we propagate both TE and TM modes. 
        outdir : str (default: ''). 
        tag : str. 
            Unique identifier for the waveguide horn. 
        
        """
        #Get the radius at the output of the horn. 
        rad = self.geometry[1][-1]

        #Get the cascaded S matrix. Note its frequency dependance
        totalSMatrix = self.computeTotalSMatrix(freq, num_modes, outdir, tag)

        #Only grab S21
        s11 = totalSMatrix['s11']
        s21 = totalSMatrix['s21']
        s22 = totalSMatrix['s22'] 

        #Compute the reflection with free space 
        imped_matrix = circ.getImpedMatrix(freq*1.0E9, num_modes, rad)

        free_space_imped = 376.730313668
        reflect_matrix = (free_space_imped - imped_matrix)/(free_space_imped + imped_matrix)
        #Note the division is element wise 
        
        #Compute the inverse 
        identity = np.identity(num_modes*2, dtype = complex)
        horn_reflect = np.linalg.inv(identity - s22@reflect_matrix) 

    
        s21_cascaded = horn_reflect@s21 

        #Grab the amplitudes of all the modes, given an excitation.
        #This excitation is purely from the fundamental first mode (TE11).
        # If we want more complex inputs (HE11 modes) we would build
        # it here (with appropriate phase delays) 
        s21_firstmode = s21_cascaded[:,0] 


#Get the EField of all the WG modes as a list of functions [TE11(rho,phi,z), TE12(rho, phi, z), ... TM11 TM12 ... ]

        #-----------------------------
        # Compute the E fields
        #  1) Get this by calling circ.TE_xy(az_num, el_num, radius, pol)[0 = x,  1 = y]. Returns an array([x,y])
        #  2) Make a big list of this. [[TE11x, TE11y],[TE12x, TE12y], ..., [TM11x, TM11y],...]
        #  3) Scale this by the mode amplitude and phase
        #  4) Add the Xs together and the Ys together
        #-----------------------------
        EField_TE_OfModes = [circ.TE_xy(1, nnum, rad, 0)[0] for nnum in np.arange(1, num_modes + 1)]
        EField_TM_OfModes = [circ.TM_xy(1, nnum, rad, 0)[0] for nnum in np.arange(1,num_modes + 1)]

        EFieldOfModes = EField_TE_OfModes + EField_TM_OfModes
        #Each element of  EFieldOfMode[mode_num] is a tuple of two matrices (E_mn,X and E_mnY) (computed at freq, rho, phi)


        #Define a function to compute the efield at a point, scaled by the amplitude. This is the contribution from each mode


        EFieldVec = lambda f, rho, phi: np.array([np.asscalar(s21_firstmode[index])*EFieldOfModes[index](f, rho, phi) for index in np.arange(0, len(EFieldOfModes))])

        
        #Sum up the vector
        #Condense all the modes so we have a vectors for E field (Ex, Ey)
        EFieldTotal = lambda rho, phi: np.sum(EFieldVec(freq*1.0E9, rho, phi),axis = 0)


        #-----------------------------
        # Compute the H field
        # The steps are identical to the comment shown above
        #-----------------------------

        HField_TE_OfModes = [circ.TE_xy(1, nnum, rad, 0)[1] for nnum in np.arange(1, num_modes + 1)]
        HField_TM_OfModes = [circ.TM_xy(1, nnum, rad, 0)[1] for nnum in np.arange(1,num_modes + 1)]
        HFieldOfModes = HField_TE_OfModes + HField_TM_OfModes

        #Define a function to compute the efield at a point, scaled by the amplitude. This is the contribution from each mode
        HFieldVec = lambda f, rho, phi: [np.asscalar(s21_firstmode[index])*HFieldOfModes[index](f, rho, phi) for index in np.arange(0, len(EFieldOfModes))]
        #Sum up the vector
        #Condense all the modes so we have a vectors for E field (Er, Ephi, Ez)
        HFieldTotal = lambda  rho, phi: np.sum(HFieldVec(freq*1.0E9, rho, phi),axis = 0)

    
        return EFieldTotal, HFieldTotal, s21_firstmode 

    def plotApertureField(self, freq, num_modes, outdir = '', tag = 'horn'):
        """
        Creates a plot of the aperture field distribution 

        Arguments
        ---------
        freq: float.
            Frequency in units of GHz.
        num_modes: int. (default: 5)
            Number of modes to include in the calculation.
            I.E. 5 modes will include 5 TE modes, and 5 TM modes
        outdir: str (default: '')
        tag: str(default: '')
            Unique identifier for this horn.
        """

        
        efield, hfield, s21_modal = self.computeApertureField(freq, num_modes, outdir, tag)
    
        
        phase = np.angle(s21_modal[0])
        phasearg = np.exp(-1j*phase)
        
        r0 = self.geometry[1][-1]
        rlist = np.linspace(1e-6, r0, 20)
        philist = np.linspace(0, 2*np.pi, 50)
        rGrid, phiGrid = np.meshgrid(rlist, philist)

        efield_grid = efield( rGrid, phiGrid)*phasearg #The vector is in X and Y
        hfield_grid = hfield( rGrid, phiGrid)*phasearg


        efield_mag = np.sqrt(np.abs(efield_grid[0])**2 + np.abs(efield_grid[1])**2)
        hfield_mag = np.sqrt(np.abs(hfield_grid[0])**2 + np.abs(hfield_grid[1])**2)



        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 5))
        ax1 = plt.subplot(121, polar = True)
        ax2 = plt.subplot(122, polar = True)
        #Create the color first
        cb1 = ax1.contourf(phiGrid, rGrid, efield_mag/np.max(efield_mag), cmap = 'autumn', vmin = 0, vmax = 1)

        ax2.contourf(phiGrid, rGrid, hfield_mag/np.max(hfield_mag), cmap = 'autumn', vmin = 0, vmax = 1)
        fig.colorbar(cb1, ticks = [0, 0.25, 0.5, 0.75, 1], ax = [ax1, ax2])

        #Create the vector plot
        '''
        ax1.streamplot(philist, rlist, #phiGrid, rGrid, 
                       np.real(efield_grid_polar[1].T), np.real(efield_grid_polar[0].T))#, width = 0.01)
        ax2.streamplot(philist, rlist,#phiGrid, rGrid, 
                       np.real(hfield_grid_polar[1].T), np.real(hfield_grid_polar[0].T))#, width = 0.01)
        '''
        #Create a quiver plot
        ax1.quiver(phiGrid, rGrid, np.real(efield_grid[0]), np.real(efield_grid[1]), width = 0.003)
        ax2.quiver(phiGrid, rGrid, np.real(hfield_grid[0]), np.real(hfield_grid[1]), width = 0.003)
      

        #Remove the radial ticks
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        
        
        ax1.set_title('E field')
        ax2.set_title('H field')
        fig.suptitle('{} Horn at {} GHz'.format(tag, freq), fontsize=16)
        plt.savefig(outdir + '{}_aperturepattern_f{}.png'.format(tag, int(freq)))

        #plt.show()
        plt.close()

        print('Saved figure at: {}{}_aperturepattern_f{}.png'.format(outdir, tag, int(freq)))


    def computeFarFieldBeam(self,freq, num_modes, outdir = '',  tag = 'horn', deltaTheta = 1*np.pi/180., deltaPhi = 5.*np.pi/180.):
        """ 
        Computes the far field beam pattern and saves the data to disk. 
        This function computes everything you would want 
        when analyzing a horn. 

        Arguments
        ---------
        freq: float.
            Frequency in units of GHz.
        num_modes: int. (default: 5)
            Number of modes to include in the calculation.
            I.E. 5 modes will include 5 TE modes, and 5 TM modes
        outdir: str (default: '')
        tag: str(default: '')
            Unique identifier for this horn.
        deltaTheta: float (radians) 
            The polar angular resolution of the beam pattern. 
        deltaPhi: float (radians) 
            The azimuthal angular resolution of the beam pattern.

        """
        c =2.99792458E8
        thetaArray = np.arange(0, np.pi/2, deltaTheta) 
        phiArray = np.arange(0, 2*np.pi, deltaPhi)
        lambda0 = c/(freq*1.0E9)
        
        filename = outdir + tag + '_beampattern_f{}ghz.npz'.format(int(freq))

        #Define a generic 2D fourier transformer 
        #Computes the 2D fourier transform in the far field at position theta,phi, given a aperture field distribution
        #############################
        # Set up the calculation 
        ############################
        ti = time.time()
        
        efield, hfield, s21_modal = self.computeApertureField(freq, num_modes, outdir, tag)


        ti = time.time()

        #We want to rotate the phasing of the electric field so the fundamental mode is in-phase 
        phase = np.angle(s21_modal[0])
        phasearg = np.exp(-1j*phase)


        aper_rad = self.geometry[1][-1] #Get the output radius
        
        #Get the sample spacing. We sample at 1/20 of wavelength.
        #Finer resolution would give us more accurate side
        #lobe calculations.
        #This resolution has a big impact on code speed. 
        deltaX = 0.05*c/(freq*1.0E9)
        deltaY = 0.05*c/(freq*1.0E9)

        #Define an XY mesh 
        xList = np.arange(-1.01*aper_rad, 1.01*aper_rad, deltaX) 
        yList = np.arange(-1.01*aper_rad, 1.01*aper_rad, deltaY) 
        xMesh, yMesh = np.meshgrid(xList, yList) 
        r0Mesh = np.sqrt(xMesh**2 + yMesh**2)
        phi0Mesh = np.arctan2(yMesh, xMesh) 
        mask = np.where(r0Mesh > aper_rad) #Create a mask for the matrix   
        #Compute the electric field, and then mask out the unphysical regions        
        e0Mesh = efield(r0Mesh, phi0Mesh)
        
        exfieldMesh = e0Mesh[0,:,:]*phasearg
        eyfieldMesh = e0Mesh[1,:,:]*phasearg

        exfieldMesh[mask] = 0.
        eyfieldMesh[mask] = 0.
               
        normFactor= np.ones(xMesh.shape, dtype = complex)
        normFactor[mask] = 0. #If you want the aperture efficiency to not fold in geometric efficiency, then you can comment this line. 
       

        #Compute the aperture efficiency
        overlap_int = np.sum(np.sum(np.abs(exfieldMesh*normFactor), axis = 0), axis = 0)
        efield_int = np.sum(np.sum(np.abs(exfieldMesh)**2 + np.abs(eyfieldMesh)**2, axis = 0), axis = 0)
        norm_int = np.sum(np.sum(np.abs(normFactor)**2, axis = 0), axis = 0)

        aper_eff = overlap_int/np.sqrt(efield_int*norm_int)



        efield_int_x = np.sum(np.sum(np.abs(exfieldMesh)**2, axis = 0), axis = 0)
        pol_eff = efield_int_x/efield_int 
        #################
        #Compute the E and H plane slices
        
        index_slice1 = int(len(yList)/2) 
        
        eslice = np.sqrt(np.abs(exfieldMesh[index_slice1,:])**2 + np.abs(eyfieldMesh[index_slice1,:])**2)
        hslice = np.sqrt(np.abs(exfieldMesh[:,index_slice1])**2 + np.abs(eyfieldMesh[:,index_slice1])**2)

        eslice_norm = eslice/np.max(eslice)
        hslice_norm = hslice/np.max(hslice)

        tf = time.time()
        print('Time to compute the aperture field {} seconds'.format(tf - ti))
        
        ti = time.time() 

        ###################
        # Compute the 2D Fourier transform over our mesh
        ###################
        thetaMesh, phiMesh, xMesh, yMesh = np.meshgrid(thetaArray, phiArray, xList, yList)
        
        sinthetaMesh = np.sin(thetaMesh)
        sinphiMesh = np.sin(phiMesh)
        cosphiMesh = np.cos(phiMesh) 


        beta = 2*np.pi/lambda0
        
        angleMesh = xMesh*sinthetaMesh*cosphiMesh + yMesh*sinthetaMesh*sinphiMesh
       

        #Holy cow I hate meshgrid, but essentially meshgrid switches the indexes order so this corrects for the different in xmesh,ymesh
        angleMesh = np.transpose(angleMesh, axes = (0, 1 , 3, 2))
        phaseMesh = np.exp(1j*beta*angleMesh)
        dim1, dim2, dim3, dim4 = phaseMesh.shape 
        #dim1 = phi dimension
        #dim2 = theta dimension
        #dim3 = X dimension
        #dim4 = Y dimension 


        exfieldMesh_adj = np.broadcast_to(exfieldMesh, (dim1, dim2, dim3, dim4))
        eyfieldMesh_adj = np.broadcast_to(eyfieldMesh, (dim1, dim2, dim3, dim4))
        
        
        
        pxMesh = np.sum(np.sum(exfieldMesh_adj*phaseMesh, axis = 2), axis = 2)
        pyMesh = np.sum(np.sum(eyfieldMesh_adj*phaseMesh, axis = 2), axis = 2)



        thetaMesh, phiMesh = np.meshgrid(thetaArray, phiArray)        
        #Turn the 2D fourier transform into a far field pattern 
        Etheta = pxMesh*np.cos(phiMesh)  + pyMesh*np.sin(phiMesh)
        Ephi = -1*pxMesh*np.sin(phiMesh)*np.cos(thetaMesh)  + pyMesh*np.cos(phiMesh)*np.cos(thetaMesh)
        Etotal = np.sqrt(np.abs(Etheta)**2 + np.abs(Ephi)**2) 

        #####################################
        # Turn this into a plot 
        X = thetaMesh*np.cos(phiMesh)*180/np.pi 
        Y = thetaMesh*np.sin(phiMesh)*180/np.pi
        Z = Etotal/np.max(Etotal)  

        #Compute the cumulative power as a function of polar angle. 
        eplane = 0 
        num_, _ = phiMesh.shape 
        eplane_index = 0 
        hplane_index = int(round(num_/4))
        pow_array = np.cumsum(np.sin(X[0,:]*np.pi/180)*Z[0,:]**2) 
        pow_array = pow_array/np.max(pow_array)



        tf = time.time()
        print('Time to compute 2d fourier transform {} sec'.format(tf -ti))



        ############################
        # Save the data 
        ############################

        np.savez(filename, 
                 freq = freq,
                 num_modes = num_modes, 
                 geometry = {'rvals_plot': self.geometry_plot[1], 
                             'zvals_plot': self.geometry_plot[0]},
                 aper_xcorr = xList, #The aperture field pattern
                 aper_ycorr = yList,
                 aper_exfield = exfieldMesh,
                 aper_eyfield = eyfieldMesh,
                 aper_eslice = eslice_norm,
                 aper_hslice = hslice_norm,
                 aper_eff = aper_eff,
                 pol_eff = pol_eff, 
                 thetaMesh = thetaMesh, #The 3D beam pattern
                 phiMesh = phiMesh, 
                 Etheta = Etheta, 
                 Ephi = Ephi, 
                 beamPattern = Etotal, 
                 X = X, 
                 Y = Y, 
                 Z = Z,
                 eplane_corr = X[eplane_index,:], #E and H slices of the pattern
                 hplane_corr = Y[hplane_index,:],
                 eplane_pow = Z[eplane_index,:]**2,
                 hplane_pow = Z[hplane_index,:]**2, 
                 eplane_pow_db = 20*np.log10(Z[eplane_index,:]),
                 hplane_pow_db = 20*np.log10(Z[hplane_index,:]),
                 deg_slice = X[eplane_index,:], #Integrated power
                 encpow = pow_array) 
        

    def summary_plots(self, freq, outdir = '', tag = 'horn'):
        """ 
        Create a summary of the horn antenna performance at a 
        given frequency. Includes the modal structure, 
        beam pattern, beam ellipticity, aperture efficiency. 

        Arguments
        ---------
        freq: float.
            Frequency in units of GHz.
        outdir: str (default: '')
        tag: str(default: '')
            Unique identifier for this horn.

        """
        #Load the data
        data_sparams = outdir + tag + '_sparams_f{}ghz.npz'.format(int(freq))
        data_beam = outdir + tag + '_beampattern_f{}ghz.npz'.format(int(freq))

        outname = outdir + 'summary_{}_f{}ghz.png'.format(tag, int(freq),)
        sparams = np.load(data_sparams, allow_pickle = True)
        beam = np.load(data_beam, allow_pickle = True)
        #####################################
        #Create the master figure handle
        ####################################
        fig = plt.figure(figsize = (10,10))
        gs =gridspec.GridSpec(3, 3, width_ratios = [1, 1, 1])
        #Create a geometry plot in the top row 
        ax0 = fig.add_subplot(gs[0, :2])


        zvals_plot = beam['geometry'].item()['zvals_plot']
        rvals_plot = beam['geometry'].item()['rvals_plot']
        ax0.plot(zvals_plot*1.0E3, rvals_plot*1.0E3, color = 'k')
        ax0.plot(zvals_plot*1.0E3, -1.0*rvals_plot*1.0E3, color = 'k')
        ax0.set_xlabel('Z-axis (mm)')
        ax0.set_ylabel('R-axis (mm)')
        ax0.set_aspect('equal')
        
        ############################
        # Create the aperture field stream plot 
        aper_rad = rvals_plot[-1] 
        ax1 = fig.add_subplot(gs[1, 0])
        aperX_corr = beam['aper_xcorr']*1.e3
        aperY_corr = beam['aper_ycorr']*1.e3
        exfieldMesh = beam['aper_exfield']
        eyfieldMesh = beam['aper_eyfield']
        ax1.streamplot(aperX_corr, aperY_corr,np.real(exfieldMesh), np.real(eyfieldMesh), color = 'k')
        #plt.title('Aperture field at {} GHz'.format(int(freq)))
        ax1.set_xlim([-aper_rad*1e3, aper_rad*1e3])
        ax1.set_ylim([-aper_rad*1e3, aper_rad*1e3])
        ax1.set_xlabel('Aperture (mm)')
        ax1.set_ylabel('Aperture (mm)')

        ###############################
        #Create the aperture field slices (E/H)
        eslice_norm = beam['aper_eslice']
        hslice_norm = beam['aper_hslice']
        aper_eff = beam['aper_eff']
        pol_eff = beam['pol_eff']
        ax2 = fig.add_subplot(gs[1, 1])
        
        ax2.plot(aperX_corr, eslice_norm, label = 'E slice', color = 'k')
        ax2.plot(aperY_corr, hslice_norm, label = 'H slice', color = 'k', linestyle = '--')
        ax2.text(aper_rad*0.65*1.0E3, 0.95, r'$\eta_{aper}$: %0.3f'%(aper_eff), ha = 'center', va = 'center', size = 10)
        ax2.text(aper_rad*0.65*1.0E3, 0.85, r'$\eta_{pol}$: %0.3f'%(pol_eff), ha = 'center', va = 'center', size = 10)

        #ax2.legend(loc = 'lower left')
        ax2.set_xlabel('Aperture position (mm)')
        ax2.set_ylabel('E field amplitude |E|')
        #plt.title('Aperture Field Cross section at {} GHz'.format(int(freq)))
        ax2.set_xlim([-aper_rad*1.0E3, aper_rad*1.0E3])
        ax2.set_ylim([0,1])
                 
        ##################################
        # Create the beam
        ax3 = fig.add_subplot(gs[2,0])
        ax4 = fig.add_subplot(gs[2,1])
        ax5 = fig.add_subplot(gs[2,2])

        deg_slice = beam['deg_slice']
        encpow = beam['encpow']
        eplane_corr = beam['eplane_corr']
        hplane_corr = beam['hplane_corr']
        eplane_pow = beam['eplane_pow']
        hplane_pow = beam['hplane_pow']
        eplane_pow_db = beam['eplane_pow_db']
        hplane_pow_db = beam['hplane_pow_db']
      
        index1 = np.where(eplane_pow < 0.5)[0][0]
        index2 = np.where(hplane_pow < 0.5)[0][0]
        efwhm = eplane_corr[index1]
        hfwhm = hplane_corr[index2]
      
        
        deltaFWHM = np.abs(efwhm - hfwhm)


        beam_ellip = deltaFWHM/(efwhm + hfwhm)
        
        ax3.plot(eplane_corr, eplane_pow, color = 'k', label= 'E plane')
        ax3.plot(hplane_corr, hplane_pow, label = 'H plane',linestyle = '--', color = 'k')
        ax3.text(8, .95, r'$\epsilon: $ %0.3f'%(beam_ellip), ha = 'center', va = 'center', size = 10) 
        ax3.legend(loc = 'lower left') 
        ax3.set_xlabel('Angle(deg)')
        ax3.set_ylabel('Beam linear scale')
        ax3.set_xlim([0, 10])
        ax3.set_ylim([0.5, 1])
        
        ax4.plot(eplane_corr, eplane_pow_db, color = 'k', label = 'E plane')
        ax4.plot(hplane_corr, hplane_pow_db, color ='k', linestyle = '--', label = 'H plane')
        #ax4.legend()
        ax4.set_xlabel('Angle (deg)')
        ax4.set_ylabel('Beam (dB)')
        ax4.set_ylim([-60, 0])
    
        ax5.plot(deg_slice, encpow, color = 'k')
        ax5.set_xlabel('Angle (deg)')
        ax5.set_ylabel('Enclosed Beam Power')
        ax5.set_xlim([0,40])

        ##########################
        # Create a summary statistics of the modes
        num_modes = sparams['num_modes'] 
        s11_block = sparams['s11']
        s21_block = sparams['s21']

        #Efficiency of transmission
        return_loss = np.abs(s11_block[0,0])**2 #Fundamental mode
        pow_trans = np.sum(np.abs(s21_block[:, 0])**2)
        ohm_loss =1 - return_loss - pow_trans
        
        
        s21 = s21_block[:, 0]

        sorted_modes_indexes = np.argsort(np.abs(s21))[::-1]
        sorted_modes = np.sort(np.abs(s21))[::-1]


        sorted_modes_label = []
        sorted_modes_amp = []
        sorted_modes_phase = []
        for i in np.arange(0, 5):
            if sorted_modes_indexes[i] < num_modes:
                label = r'TE$_{1' + str(sorted_modes_indexes[i] + 1) + '}$'
                
                 
            else:
                label = r'TM$_{1' + str(sorted_modes_indexes[i] + 1 - num_modes) + '}$'
 

            sorted_modes_label.append(label)
            sorted_modes_amp.append(np.abs(s21[sorted_modes_indexes[i]])**2)
            sorted_modes_phase.append(np.angle(s21[sorted_modes_indexes[i]], deg = True))
            
        output_text = []
        first_line ='Output modes $|s_{21}|^2 < \phi $:' 
        output_text.append(first_line) 
        for i in np.arange(0, 5): 
            next_line = '{}: {} $ < $ {}'.format(sorted_modes_label[i],
                                                     '%0.3f'%sorted_modes_amp[i],
                                                     '%0.3f'%sorted_modes_phase[i])
            output_text.append('\n \t')
            output_text.append(next_line)


        output_text = ' '.join([str(elem) for elem in output_text])
        

        
        axtext = fig.add_subplot(gs[1, -1])
        input_text = 'Input modes: \n \t ' + r'TE$_{11}$' + ' 1.0 $<$ 0'

        axtext.text(0.05, 0.95, 'Num modes: {}'.format(2*num_modes), ha = 'left', va = 'top', size = 10)
        axtext.text(0.05, 0.70, r'Return loss $(|s_{11}|^2): $ %0.3f'%(return_loss), ha = 'left', va = 'top', size = 10)
        axtext.text(0.05, 0.85, input_text, ha = 'left', va = 'top', size = 10)
        axtext.text(0.05 ,0.64, output_text, ha = 'left', va = 'top', size = 10)

        if not np.isinf(self.conductivity):
            axtext.text(0.05, 0.20, r'$\sigma$: %0.2e S/m'%(self.conductivity), ha = 'left', va = 'top', size = 10)
            axtext.text(0.05, 0.15, 'Ohmic loss: %0.3f '%(ohm_loss), ha = 'left', va = 'top', size = 10)
        axtext.text(0.05, 0.08, 'Optical eff: %0.3f'%(pow_trans), ha = 'left', va = 'top', size = 12) 
        axtext.set_xticks([])
        axtext.set_yticks([]) 

                        

        
        #plt.tight_layout()
        
        fig.suptitle('{} Horn at {} GHz'.format(tag, freq), fontsize=16)
        fig.tight_layout(rect= [0, 0, 1, 0.97])
        plt.savefig(outname) 
        
        plt.show()
        plt.close() 
        
