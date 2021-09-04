import numpy as np 
import matplotlib.pyplot as plt
""" 
HornGeometry.py 

Contains a few presets of common horn types. 
This is far from a complete list.  

"""

def gaussian_corrugated_wg(r0, alpha, lam, rend, taper_len = 0,fin_width = 1., corr_width = 1., straight_sec = 0.001, units = 'm'):
    """
    Creates a gaussian tapered horn with corrugations. 
    
    Arguments 
    ---------
    r0 : float (inner radius of the horn). 
    alpha: float (radians). Describes the angle of the horn.
    lam: float. The wavelength of the horn. 
         This determines the corrugation depth. 
    rend : float. The outer radius of the horn. 
    taper_len : float (in meters). The  length of the waveguide taper. 
            This length mixes TE11 -> HE11
    fin_width : float (in lambda/4 units).
            The width of the fins. 
    corr_width : float (in lambda /4 units). 
            The width of the troughs.
    straight_sec : float. Default 1mm. 
            The beginning waveguide length. 
    units: str. Default 'm'. The units associated with all the length
            elements. HornMM expects units in meters. 
    """

    # Example, try 
    #r0 = 0.676
    #alpha = 2.0
    #f0 = 150.E9
    #lam = 3.0E8/f0*1.0E3
    #taperlen = 3.
    #rend = 7.5
    #fin_width = 0.5
    #corr_width = 0.5 
    #
    if units == 'm':
        pass
    elif units == 'cm':
        r0 = r0*1e-2
        lam = lam*1e-2
        rend = rend*1e-2
        taper_len = taper_len*1e-2
    
    elif units == 'mm':
        r0 = r0*1e-3
        lam = lam*1e-3
        rend = rend*1e-3
        taper_len = taper_len*1e-3
                                        
    else:
        print('Unknown units. Quitting')
        exit()
                                

    rad_array = []
    height_array = []

    rad_func = lambda z: r0*np.sqrt(1 + (lam*z/(np.pi*alpha**2*r0**2))**2)

    #Total length
    totallen = (np.pi*alpha**2*r0**2)/lam*np.sqrt((rend/r0)**2 - 1)

    corrA = lam/4
    corrB = lam/2
    len_elem = corrA


    len_elem = corrA

    len_fin = fin_width*corrA
    len_corr = corr_width*corrA
    len_period = len_fin + len_corr
    num_elem = np.floor(totallen/len_period)
      
    for i in np.arange(0, num_elem):
        #Determine where we are in (r,z)
        zval = i*len_period
        rval = rad_func(zval)
        #Figure out the taper height
        if zval > taper_len:
            fin_height = corrA
        else:
            fin_height = (corrA - corrB)/taper_len*zval + corrB

        rad_array.append(rval)
        height_array.append(len_fin)

        rad_array.append(rval + fin_height)
        height_array.append(len_corr)


    rad_array.append(rend)
    height_array.append(len_fin)

    if straight_sec > 0:
        rad_array = np.insert(rad_array, 0, r0)
        height_array = np.insert(height_array, 0, straight_sec)
    return rad_array, height_array, totallen




def conical_wg(num_elems, rlist, zlist, straight_sec = 0.001, units = 'm'):
    """
    Creates a smooth sided conical waveguide horn
    
    Arguments
    ---------
    num_elems : int. Number of waveguide sectiotns per conical section. 
    rlist: array of floats. An array outlining radius 
        of each cone element. 
    zlist: array of floats. An array outlining the heights
        of each cone element. 
    straight_sec : float. Default 1mm.
        The beginning waveguide length.
    units: str. Default 'm'. The units associated with all the length
            elements. HornMM expects units in meters.
    """
      

    rlist = np.array(rlist)
    zlist = np.array(zlist)
    if units == 'm':
        pass
    elif units == 'cm':
        rlist = rlist*1e-2
        zlist = zlist*1e-2
    elif units == 'mm':
        rlist = rlist*1e-3
        zlist = zlist*1e-3
    else:
        print('Unknown units. Quitting')
        exit()
            
    rarray = []
    zarray = []
            
    rarray.append(rlist[0])
    zarray.append(zlist[0])


    num_segs = len(zlist)

        
    for next_seg in np.arange(1, num_segs):
        running_r = rarray[-1]
        running_z = zarray[-1]
        zcorr = np.linspace(running_z, running_z + zlist[next_seg], num_elems)
        zdiff = np.diff(zcorr)
        rcorr = np.linspace(running_r, rlist[next_seg], num_elems)
        
        rarray = np.concatenate((rarray, rcorr[1:]))
        zarray = np.concatenate((zarray,zdiff))

    if straight_sec > 0:
        rarray = np.insert(rarray,0, rlist[0])
        zarray = np.insert(zarray, 0, straight_sec) 
        
    totallen = np.sum(zlist)
    return rarray, zarray, totallen

def cubic_spline_wg(num_elems, r_nodes, z_nodes, bc_type = 'natural', straight_sec = 0.001, units = 'm'):
    """
    Creates a cubic spline waveguide 
    
    Arguments
    ---------
    num_elems : int. Number of waveguide sectiotns per conical section. 
    r_nodes: array of floats. 
    z_nodes: array of floats. 
         r_nodes and z_nodes indicate pivot points for the spline. 
    bc_type : 'natural' or 'clamped'.
         'natural' indicates a 2nd deriv = 0. 
         'clamped' indicates a 1st deriv = 0. 
    straight_sec : float. Default 1mm.
        The beginning waveguide length.
    units: str. Default 'm'. The units associated with all the length
            elements. HornMM expects units in meters.
    """

    from scipy.interpolate import CubicSpline
    
    rlist = np.array(r_nodes)
    zlist = np.array(z_nodes)
    if units == 'm':
        pass
    elif units == 'cm':
        rlist = rlist*1e-2
        zlist = zlist*1e-2
    elif units == 'mm':
        rlist = rlist*1e-3
        zlist = zlist*1e-3
    else:
        print('Unkown units. Quitting')
        exit()
    

    f = CubicSpline(zlist, rlist, bc_type = bc_type)
    zarray = np.linspace(zlist[0], zlist[-1], endpoint = True, num = num_elems)
    
    rarray = f(zarray)
    
    #Turn this into sections 
    zarray = np.diff(zarray)

    if straight_sec > 0:
        rarray = np.insert(rarray, 0, rlist[0])
        zarray = np.insert(zarray, 0, straight_sec)
        
        totallen = np.sum(zarray)
    return rarray[:-1], zarray, totallen

                                                        
def hermite_spline_wg(num_elems, r_nodes, z_nodes, drdz_nodes, straight_sec = 0.001, units = 'm'):
    """
    Creates a hremite spline waveguide 

    Arguments
    ---------
    num_elems : int. Number of waveguide sectiotns per conical section.
    r_nodes: array of floats.
    z_nodes: array of floats.
         r_nodes and z_nodes indicate pivot points for the spline.
    drdz_nodes: array of floats. 
         indicates the angle at which the spline should be at the 
         respective nodes. 
    straight_sec : float. Default 1mm.
        The beginning waveguide length.
    units: str. Default 'm'. The units associated with all the length
            elements. HornMM expects units in meters.
    """
   
    from scipy.interpolate import CubicHermiteSpline

    # straight_sec is the length from the end of the horn to the detector
    rlist = np.array(r_nodes)
    zlist = np.array(z_nodes)
    drdzlist = np.array(drdz_nodes)
    if units == 'm':
        pass
    elif units == 'cm':
        rlist = rlist*1e-2
        zlist = zlist*1e-2
        drdzlist = drdzlist*1e-2
    elif units == 'mm':
        rlist = rlist*1e-3
        zlist = zlist*1e-3
        drdzlist = drdzlist*1e-3
    else:
        print('Unknown units. Quitting')
        exit()
        
    rarray = []
    zarray = []
    drdzarray = []

    #Create the interpolation
    f = CubicHermiteSpline(rlist, zlist, drdzlist)
    
    rarray = np.linspace(rlist[0], rlist[-1], endpoint = True, num  = num_elems)
    zpos = f(rarray)

    zarray = np.diff(zpos) 

    if straight_sec > 0:
        rarray = np.insert(rarray, 0, rlist[0]) 
        zarray = np.insert(zarray, 0, straight_sec)
    
    totallen = np.sum(zarray) 
    return rarray[:-1], zarray, totallen
                                                                        
