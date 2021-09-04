import numpy as np
from scipy.special import jv
from scipy.special import jvp
from scipy import special
from scipy.integrate import dblquad
from numpy.lib import scimath
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

""" 
CircWGNorm.py  

Script that contains all the functions relating to circular waveguides. 
In this code we use the convention that 
TEnm where n is the azimuthal number 
     where m is the radial number 
This is to be consistent with Pozar's definition of TE/TM modes. 

"""

#####################
# Define some fundamental constants
c = 2.99792458e8 #Speed of light m/s
eta = 376.730313668 #Impedance of free space 
mu = 1.2566e-6 #In Si m*kg/C^2
epsilon = 8.854e-12 #F/m
    

#These return the zeros of the bessel functions or the derivative of the bessel function 
def p(n,m):
    """ 
    The zeros of the bessel function of the first kind J(n,m)
    """
    besselN_zeros = special.jn_zeros(n,m)
    return besselN_zeros[m - 1]
def pp(n,m):
    """ 
    The zeros of the deriv of the bessel function of the first kind J'(n,m)
    """
    besselpN_zeros =  special.jnp_zeros(n,m)
    return besselpN_zeros[m - 1]

def getModeRoot(mode):
    """ 
    Get the root of a given mode.
    """
    mtype = mode['mtype']
    n = mode['azimuth']
    m = mode['radial']
    if mtype == 'TE':
        return pp(n,m)
    else:
        return p(n,m)
    

def integralJJ(n, a,b,x0):
    """ 
    Takes the integral 
      int drho rho*Jv(a*rho)Jv(b*rho)  from 0 to x0. 
    
    This integral is very useful for normalization 
    """
    if a == b:
        total = integralNorm(n, a, x0)
    else: 
        num = b*jv(n - 1,b*x0)*jv(n, a*x0) - a*jv(n - 1, a*x0)*jv(n, b*x0)
        factor = x0/(a**2 - b**2)
        total = num*factor
    return total


def integralNorm(n, alpha, x0):
    """ 
    Takes the integral 
       int drho rho*Jn(alpha*rho)^2 from rho = 0 to x0
    """
    num = alpha*jv(n - 1, alpha*x0)**2 - 2*n/x0*jv(n - 1, alpha*x0)*jv(n, alpha*x0) + alpha*jv(n, alpha*x0)**2
    factor = x0**2/(2*alpha)

    return num*factor

def impedTE(n,m,a):
    """
    Impedance of a TEnm mode with radius a 
    """
    k = lambda f:2*np.pi*f/c
    kc = pp(n,m)/a
    beta = lambda f: scimath.sqrt(np.power(k(f),2) - np.power(kc,2))
    imped = lambda f:k(f)*eta/beta(f)
    return imped

def impedTM(n,m,a):
    """
    Impedance of a TMnm mode with radius a
    """
    k = lambda f: 2*np.pi*f/c
    kc = p(n,m)/a
    beta = lambda f: scimath.sqrt(np.power(k(f),2) - np.power(kc,2))
    imped = lambda f: beta(f)*eta/k(f)

    return imped

def getImped(mode):
    """ 
    Function that spits out the impedance of a mode 
    """
    n = mode['azimuth']
    m = mode['radial']
    a = mode['rad']

    if mode['mtype'] == 'TE':
        imped = impedTE(n,m,a)
    else:
        imped = impedTM(n,m,a)

    return imped

#Return the  normalization factor, unadjusted for impedance 
def normTE(n,m,a):
    """ 
    Normalization factor for a TEmn mode with radius a 
    """
    integral_factor = integralNorm(n + 1, pp(n,m)/a,a) + integralNorm(n - 1, pp(n,m)/a,a)
    norm_squared = np.pi/2*integral_factor
    imped = impedTE(n,m,a)

    norm = lambda f: scimath.sqrt(np.abs(imped(f))/norm_squared)
    
    return norm

def normTM(n,m,a):
    """ 
    Normalization factor for TMnm mode with radius a
    """
    integral_factor = integralNorm(n + 1, p(n,m)/a,a) + integralNorm(n - 1, p(n,m)/a,a)
    norm_squared = np.pi/2*integral_factor
    imped = impedTM(n,m,a)
    norm = lambda f: scimath.sqrt(np.abs(imped(f))/norm_squared)
    
    return norm

def getNorm(mode):
    """ 
    Get the correct normalization factor for a mode. 
    """
    n = mode['azimuth']
    m = mode['radial']
    a = mode['rad']
    if mode['mtype'] == 'TE':
        norm = normTE(n,m,a)
    else:
        norm = normTM(n,m,a)

    return norm

#Returns a vector function TEnm mode in a circular waveguide with radius a ati polarization (polar = 0). polar is equivalent to B in pozar: (Er, Etheta, Ez) all as a function of frequency, rho, theta.
def TE(n,m,a, polar = 0):
    """ 
    TEnm mode with radius a 
    Returns 
    -------
    (Evec, Hvec) a tuple of Evec and Hvec which are vector fields. 
                 Evec/Hvec are in cylindrical coordinates (f, rho, phi). 
    
    Arguments 
    ---------
    n - int. Azimuthal number. 
    m - int. Radial number. 
    a - float. Radius of waveguide. 
    polar - float. Polarization state. 
    """
    omega = lambda f: 2*np.pi*f
    kc = pp(n,m)/a #Cutoff wavenumber in guide
    k = lambda f: omega(f)/c #Wavenumber in free space
            
    beta = lambda f: scimath.sqrt(np.power(k(f),2) - np.power(kc,2)) #Wavespeed of TEmn mode
    
    imped = lambda f: k(f)*eta/beta(f)
    #Polarization state
    A = 1 - polar
    B = polar
    
    norm =  normTE(n,m,a)
        
    Erho = lambda f, rho, phi: -1j*norm(f)*(A*np.cos(n*phi) - B*np.sin(n*phi))*n*jv(n, kc*rho)/(kc*rho)
    Ephi = lambda f, rho, phi: 1j*norm(f)*(A*np.sin(n*phi) + B*np.cos(n*phi))*jvp(n,kc*rho)
    Ez = 0
    
    Evec = lambda f, rho, phi: np.array([Erho(f, rho, phi), Ephi(f, rho, phi)])#, Ez])
        
    Hrho = lambda f, rho, phi: -1j*norm(f)/imped(f)*(A*np.sin(n*phi) + B*np.cos(n*phi))*jvp(n,kc*rho)
    Hphi = lambda f, rho, phi: -1j*norm(f)/imped(f)*(A*np.cos(n*phi) - B*np.sin(n*phi))*n*jv(n,kc*rho)/(kc*rho)
    Hz = lambda f, rho, phi: (A*np.sin(n*phi) + B*np.cos(n*phi))*jv(n, kc*rho)
    
    Hvec = lambda f, rho, phi: np.array([Hrho(f, rho, phi), Hphi(f, rho, phi)])

    
    
    return (Evec, Hvec)
    
def TE_xy(n,m,a,polar = 0):
    """ 
    TEnm  mode with radius a. 
    Returns 
    -------
    (Evec, Hvec) a tuple of Evec and Hvec which are vector fields. 
                 Evec/Hvec are in cartesian coordinates (f,x,y).
    
    Arguments 
    ---------
    n - int. Azimuthal number. 
    m - int. Radial number. 
    a - float. Radius of waveguide. 
    polar - float. Polarization state. 
    """
    omega = lambda f: 2*np.pi*f
    kc = pp(n,m)/a #Cutoff wavenumber in guide
    k = lambda f: omega(f)/c #Wavenumber in free space
    
    beta = lambda f: scimath.sqrt(np.power(k(f),2) - np.power(kc,2)) #Wavespeed of TEmn mode
    imped = lambda f: k(f)*eta/beta(f)
    #Polarization state
    A = 1 - polar
    B = polar
    
    norm =  normTE(n,m,a)
    Ex = lambda f, rho, phi: 0.5*norm(f)*(
        jv(n + 1, kc*rho)*(A*np.cos((n + 1)*phi) +-1*B*np.sin((n + 1)*phi) ) +
        jv(n - 1, kc*rho)*(A*np.cos((n - 1)*phi) + -1*B*np.sin((n - 1)*phi)))
    Ey = lambda f, rho, phi: 0.5*norm(f)*(
        jv(n + 1, kc*rho)*(A*np.sin((n + 1)*phi) + B*np.cos((n + 1)*phi)) -
        jv(n - 1, kc*rho)*(A*np.sin(( n - 1)*phi) + B*np.cos((n - 1)*phi)))
    Exy = lambda f, rho, phi: np.array([Ex(f, rho, phi), Ey(f, rho, phi)])

    
    Hx = lambda f, rho, phi: 0.5*norm(f)/imped(f)*(
        -1*jv(n + 1, kc*rho)*(A*np.sin((n + 1)*phi) + B*np.cos((n + 1)*phi)) + 
        jv(n - 1, kc*rho)*(A*np.sin((n - 1)*phi) + B*np.cos((n - 1)*phi)))
    
    Hy = lambda f, rho, phi: 0.5*norm(f)/imped(f)*(
        jv(n + 1, kc*rho)*(A*np.cos((n + 1)*phi) - B*np.sin((n + 1)*phi)) +
        jv(n - 1, kc*rho)*(A*np.cos((n - 1)*phi) - B*np.sin((n - 1)*phi)))
    Hxy = lambda f, rho, phi: np.array([Hx(f, rho, phi), Hy(f, rho, phi)])



    return (Exy, Hxy)

    

def TM(n,m,a,polar = 0):
    """ 
    TMnm  mode with radius a. 
    Returns 
    -------
    (Evec, Hvec) a tuple of Evec and Hvec which are vector fields. 
                 Evec/Hvec are in cylindrical coordinates (f,rho, phi).
    
    Arguments 
    ---------
    n - int. Azimuthal number. 
    m - int. Radial number. 
    a - float. Radius of waveguide. 
    polar - float. Polarization state. 
    """

    omega = lambda f: 2.*np.pi*f

    kc = p(n,m)/a #Cutoff wavenumber in guide
    k = lambda f: omega(f)/c #Wavenumber in free space
     
    beta = lambda f: scimath.sqrt(np.power(k(f),2) - np.power(kc,2)) #Wavespeedof TEmn mode
    imped = lambda f: beta(f)*eta/k(f)
    
    #Polarization state
    A = 1 - polar
    B = polar

    norm = normTM(n,m,a)
    
    Erho = lambda f, rho, phi: -1j*norm(f)*(A*np.sin(n*phi) + B*np.cos(n*phi))*jvp(n,kc*rho)
    Ephi = lambda f, rho, phi: -1j*norm(f)*(A*np.cos(n*phi) - B*np.sin(n*phi))*n*jv(n,kc*rho)/(kc*rho)
    
    Ez = lambda f, rho, phi: (A*np.sin(n*phi) + B*np.cos(n*phi))*jv(n,kc*rho)
    Evec = lambda f,rho, phi: np.array([Erho(f, rho, phi), Ephi(f, rho, phi)])#, Ez(f, rho, phi)])
    
    Hrho = lambda f, rho, phi: 1j*norm(f)/imped(f)*(A*np.cos(n*phi) - B*np.sin(n*phi))*n*jv(n,kc*rho)/(kc*rho)
    Hphi = lambda f, rho, phi: -1j*norm(f)/imped(f)*(A*np.sin(n*phi) + B*np.cos(n*phi))*jvp(n,kc*rho)
    Hz = 0
    Hvec = lambda f,rho, phi: np.array([Hrho(f,rho,phi), Hphi(f,rho,phi)])#, Hz])

    
    
    return (Evec, Hvec)


def TM_xy(n,m, a, polar = 0):
    """
    TMnm  mode with radius a.
    Returns
    -------
    (Evec, Hvec) a tuple of Evec and Hvec which are vector fields.
                 Evec/Hvec are in cartesian coordinates (f,x,y).

    Arguments
    ---------
    n - int. Azimuthal number.
    m - int. Radial number.
    a - float. Radius of waveguide.
    polar - float. Polarization state.
    """

 
    omega = lambda f: 2*np.pi*f
    
    kc = p(n,m)/a #Cutoff wavenumber in guide
    k = lambda f: omega(f)/c #Wavenumber in free space
    
    beta = lambda f: scimath.sqrt(np.power(k(f),2) - np.power(kc,2)) #Wavespeed of TEmn mode
    imped = lambda f: beta(f)*eta/k(f)
    #Polarization state
    A = 1 - polar
    B = polar
    
    norm =  normTM(n,m,a)
    Ex = lambda f, rho, phi: 0.5*norm(f)*(
        -jv(n + 1, kc*rho)*(
            A*np.cos((n + 1)*phi) + B*np.sin((n + 1)*phi)) +
        jv(n - 1, kc*rho)*(
            A*np.cos((n - 1)*phi) + B*np.sin((n - 1)*phi)))
    Ey = lambda f, rho, phi: 0.5*norm(f)*(
        jv(n + 1, kc*rho)*(-1*A*np.sin((n + 1)*phi) + B*np.cos((n + 1)*phi)) +
        jv(n - 1, kc*rho)*(-1*A*np.sin((n - 1)*phi) + B*np.cos((n - 1)*phi)))
    Exy = lambda f, rho, phi: np.array([Ex(f,rho,phi), Ey(f,rho,phi)])

    Hx = lambda f, rho, phi: 0.5*norm(f)/imped(f)*(
        jv(n + 1, kc*rho)*(A*np.sin((n + 1)*phi) - B*np.cos((n + 1)*phi)) +
        jv(n - 1, kc*rho)*(A*np.sin((n - 1)*phi) - B*np.cos((n - 1)*phi)))
    Hy = lambda f, rho, phi: 0.5*norm(f)/imped(f)*(
        -1*jv(n + 1, kc*rho)*(A*np.cos((n + 1)*phi) + B*np.sin((n + 1)*phi)) +
        jv(n - 1, kc*rho)*(A*np.cos((n - 1)*phi) + B*np.sin(( n - 1)*phi)))
    Hxy = lambda f, rho, phi: np.array([Hx(f, rho, phi), Hy(f,rho,phi)])

    return (Exy, Hxy)
    

def getImpedMatrix(freq, num_modes,a):
    """ 
    Returns a diagonal matrix (2n x 2n) with the elements being the 
    impedance of each of the modes. 
    
    Arguments 
    ---------
    freq - float (in GHz) 
    num_modes - int number of radial modes. 
    a - float (in meters) radius of the waveguide. 
    
    """
    k = lambda f: 2*np.pi*f/c
    imped_matrix = eta*np.ones((num_modes*2, num_modes*2), dtype = complex)
    for index in np.arange(0, num_modes*2):
        if index < num_modes:
            kc = pp(1,index + 1)/a #This is TE Mode
            beta = lambda f: scimath.sqrt(np.power(k(f),2) - np.power(kc,2))
            imped = lambda f: k(f)*eta/beta(f)
        else:
            kc = p(1, index + 1 - num_modes)/a #This is TM mode
            beta = lambda f: scimath.sqrt(np.power(k(f),2) - np.power(kc,2))
            imped = lambda f: beta(f)*eta/k(f)
            
        imped_matrix[index, index] = imped(freq)

    return np.array(imped_matrix)

def getBetaElls(freq, a, length, num_modes, conductivity = np.inf):
    """ 

    Returns a diagonal matrix (2n x 2n) with a phase shift 
    appropriate to each mode when traveling through a wg 
    of radius a 
    Returns
    -------
    A (2n x 2n) matrix describing the attenuation+phase shift.
    where the (n x n) first block is the TE matrix 
          the second (n x n) block is the TM matrix 
    Arguments
    ---------
    freq - float (in GHz)
    a - float (in meters) radius of the waveguide.
    length - float (in meters) length of waveguide. 
    num_modes - int. Number of modes to propagate. 
    conductivity - float (default: np.inf). Conductivity 
          of the guide in S/m. 
    """
    omega = lambda f: 2*np.pi*f
    kc_tm = lambda n,m:  p(n,m)/a
    kc_te = lambda n,m: pp(n,m)/a
    k = lambda f: omega(f)/c
    beta_tm = lambda f, n,m: scimath.sqrt(np.power(k(f), 2) - np.power(kc_tm(n,m),2))
    beta_te = lambda f, n,m: scimath.sqrt(np.power(k(f), 2) - np.power(kc_te(n,m),2))


    #Apply a surface resistance attenuation factor
    if np.isinf(conductivity):
        alpha_te = lambda f,n,m: 0.
        alpha_tm = lambda f,n,m: 0. 

    else:
        Rs = lambda f: scimath.sqrt(2*np.pi*f*mu/(2*conductivity))
        alpha_te = lambda  f, n,m: Rs(f)/(a*k(f)*eta*beta_te(f,n,m))*(kc_te(n,m)**2 + k(f)**2*m**2/(pp(n,m)**2 - m**2))

        alpha_tm  = lambda f,n,m: Rs(f)*k(f)/(a*eta*beta_tm(f,n,m))

        
    betaell_matrix = np.zeros((num_modes*2, num_modes*2), dtype = complex)
    for index in np.arange(1, num_modes*2 + 1):
        if index <= num_modes:
            coeff = np.exp((-alpha_te(freq, 1, index) + 1j*beta_te(freq, 1, index))*length)
            betaell_matrix[index - 1, index - 1] = coeff
        else:
            coeff = np.exp((-alpha_tm(freq, 1, index - num_modes) + 1j*beta_tm(freq, 1, index - num_modes))*length)
            betaell_matrix[index - 1, index - 1] = coeff

    return np.array(betaell_matrix)


#Self coupling on the left side of the guide.
def computeRmn(f, mode1):
    """ 
    Self coupling integral for mode nm  
    Returns 
    -------
    Reflection of a mode at the left of the junction

    Arguments
    ---------
    freq - float(in GHz). 
    mode1 - dict (of a given mode TE/TM nm, in a wg of radius a).
    """

    impedR_func = getImped(mode1)
    impedR = impedR_func(f)
    if isinstance(impedR, complex):
        return np.abs(impedR)/np.conj(impedR)
    else:
        return 1

def computeQmn(f, mode2):
    """ 
    Self coupling integral for mode nm  
    Returns 
    -------
    Reflection of a mode at the right of the junction

    Arguments
    ---------
    freq - float(in GHz). 
    mode2 - dict (of a given mode TE/TM nm, in a wg of radius a).
    """
    
    impedQ_func = getImped(mode2)
    impedQ = impedQ_func(f)
    if isinstance(impedQ, complex):
        return np.abs(impedQ)/np.conj(impedQ)
    else:
        return 1
    

def computePmn(freq, mode1, mode2):
    """ 
    Mutual coupling integral for mode1 nm, with mode2 n'm'  
    Returns 
    -------
    Reflection of a mode at the left of the junction

    Arguments
    ---------
    freq - float(in GHz). 
    mode1 - dict (of a given mode TE/TM nm, in a wg of radius a).
    mode2 - dict (of second TE/TM mode).
    """
    
    #Do the first check if the polarizations are different, then
    # never any coupling 
    if mode1['pol'] != mode2['pol']:
        return 0


    #We ALWAYS integrate over the smaller waveguide diameter 
    if mode2['rad'] < mode1['rad']:
        temp = mode1
        mode1 = mode2
        mode2 = temp
    
    #Get the mode parameters 
    n1 = mode1['azimuth']
    m1 = mode1['radial']
    r1 = mode1['rad']

    n2 = mode2['azimuth']
    m2 = mode2['radial']
    r2 = mode2['rad']
    #Transalte to the problem 
    f = freq
    n = n1
    mode1_root = getModeRoot(mode1)
    mode2_root = getModeRoot(mode2)

    #Get the normalization 
    norm1 = getNorm(mode1)
    norm2 = getNorm(mode2)
    imped2 = getImped(mode2)
    

    #If TE->TE or TM->TM
    if mode1['mtype'] == mode2['mtype']: 
        integral1 = integralJJ(n + 1, mode1_root/r1, mode2_root/r2, r1)
        integral2 = integralJJ(n - 1, mode1_root/r1, mode2_root/r2, r1)

        overlap_integral = np.pi/2*(integral1 + integral2)
        #Fold in the normalization
        total_integral = overlap_integral*norm1(f)*norm2(f)/np.conj(imped2(f))
        return total_integral
    #If TE -> TM
    elif mode1['mtype'] == 'TE' and mode2['mtype'] == 'TM':
        bessels = jv(n1, pp(n1, m1))*jv(n2, p(n2, m2)*r1/r2)
        overlap_integral = n1*r1*r2/(pp(n1,m1)*p(n2,m2))*bessels
        total_integral = np.pi*norm1(f)*norm2(f)/np.conj(imped2(f))*overlap_integral

        return total_integral
    #If TM -> TE
    else:
        return 0    


def calculatePMatrix(freq, num_modes, rad1, rad2):
    """    
    Computes the coupling matrix across one waveguide junction 
 
    Returns
    -------
    Returns a matrix (2num_modes x 2num_modes), with the following blocks 
    TE->TE, TM->TE 
    TE->TM, TM->TM

    Arguments
    ---------
    freq - float (in GHz)
    num_modes - int number of radial modes.
    rad1 - float (in meters) radius of the waveguide.
    
    """
    pmatrix = np.zeros((num_modes*2 , num_modes*2), dtype = complex) #modes for TE_1n and TM_1m
    
    #Loop through the number of modes
    for index in np.arange(1, num_modes*2 + 1):
        if index <= num_modes: 
            mode1 = {'mtype':'TE',
                     'azimuth': 1,
                     'radial': index,
                     'rad': rad1,
                     'pol':0}
        else:
            mode1 = {'mtype':'TM',
                     'azimuth':1,
                     'radial': index - num_modes,
                     'rad': rad1,
                     'pol':0}
        for jndex in np.arange(1, num_modes*2 + 1):
            
            if jndex <= num_modes: 
                mode2 = {'mtype':'TE',
                         'azimuth':1,
                         'radial': jndex,
                         'rad': rad2,
                         'pol':0}
            else:
                mode2 = {'mtype':'TM',
                         'azimuth':1,
                         'radial':jndex - num_modes,
                         'rad': rad2,
                         'pol':0}

                
            pmatrix[jndex - 1, index - 1] = computePmn(freq, mode1, mode2)

    return np.array(pmatrix)

def calculateRMatrix(freq, num_modes, rad1, rad2):
    """    
    Computes the self coupling matrix (reflection) 
    on the left side of the junction.
 
    Returns
    -------
    Returns a reflection  matrix (2num_modes x 2num_modes), 
    with the following blocks 
    TE->TE, TM->TE 
    TE->TM, TM->TM

    Arguments
    ---------
    freq - float (in GHz)
    num_modes - int number of radial modes.
    rad1 - float (in meters) radius of the left junction.
    rad2 - float (in meters) radius of the right junction. 
    """


    rmatrix = np.zeros((num_modes*2, num_modes*2), dtype = complex)
    for index in np.arange(1, num_modes*2 + 1):
        if index <= num_modes:
            mode1 = {'mtype':'TE',
                     'azimuth':1,
                     'radial': index,
                     'rad': rad1,
                     'pol': 0}
        else:
            mode1 = {'mtype':'TM',
                     'azimuth':1,
                     'radial':index - num_modes,
                     'rad':rad1,
                     'pol':1}
        rmatrix[index -1, index - 1] = computeRmn(freq, mode1)
    
    return np.array(rmatrix)
def calculateQMatrix(freq, num_modes, rad1, rad2):
    """    
    Computes the self coupling matrix (reflection) 
    across on the right side of the junction.
 
    Returns
    -------
    Returns a reflection  matrix (2num_modes x 2num_modes), 
    with the following blocks 
    TE->TE, TM->TE 
    TE->TM, TM->TM

    Arguments
    ---------
    freq - float (in GHz)
    num_modes - int number of radial modes.
    rad1 - float (in meters) radius of the left junction.
    rad2 - float (in meters) radius of the right junction. 
    """

    qmatrix = np.zeros((num_modes*2 , num_modes*2), dtype = complex) #modes for TE_1n and TM_1m

    for index in np.arange(1, num_modes*2 + 1):
        if index <= num_modes:
            mode2 = {'mtype':'TE',
                     'azimuth': 1,
                     'radial': index,
                     'rad': rad2,
                     'pol':0}
        else:
            mode2 = {'mtype':'TM',
                     'azimuth':1,
                     'radial': index - num_modes,
                     'rad': rad2,
                     'pol':1}
            
        qmatrix[index - 1, index - 1] = computeQmn(freq,mode2)
                     
    return np.array(qmatrix)

def calculateSMatrix(freq, num_modes, rad1, rad2):
    """ 
    Computes the scattering matrix for a waveguide interface 
    
    Returns
    -------
    A block matrix consisting of s11, s12, s21, s22 subblocks 
    for a total of a (4n x 4n) matrix where n = num_modes. 
    Each block is (2n x 2n) accounting for TE and TM modes. 
    
    Arguments
    ---------
    freq - float (in GHz). 
    num_modes - int. Number of radial modes for both TE+TM 
    rad1 - float (in meters). Radius of the left junction. 
    rad2 - float (in meters). Radius of the right junction.
    
    """
    pm = calculatePMatrix(freq, num_modes, rad1,rad2)
    rm = calculateRMatrix(freq, num_modes, rad1,rad2)
    qm = calculateQMatrix(freq, num_modes, rad1,rad2)
    
    
    if rad1 <= rad2:
        qmi = np.linalg.inv(qm)

        rmi = np.linalg.inv(np.conjugate(rm))
        pmh = np.conj(pm).T#.getH() #getH() is python2
        s11 = np.linalg.inv(np.conjugate(rm) + pmh@qmi@pm)@(np.conjugate(rm) - pmh@qmi@pm)
        s21 = 2*np.linalg.inv(qm + pm@rmi@pmh )@pm
        
        s12 = 2*np.linalg.inv(np.conjugate(rm) + pmh@qmi@pm)@ pmh
        s22 = np.linalg.inv(qm + pm@rmi@ pmh)@(-qm + pm@rmi@pmh)
        
    else:
        #Transpose the p
        pm = pm.transpose()
        qmc = np.conjugate(qm)
        qmi = np.linalg.inv(np.conjugate(qm))
        rmi = np.linalg.inv(rm)
        pmh = np.conj(pm).T #.getH()
        
        s11 = np.linalg.inv(rm + pm@qmi@pmh)@(-rm +pm@qmi@pmh)

        s12 = 2*np.linalg.inv(rm + pm@qmi@pmh)@pm
        s21 = 2*np.linalg.inv(qmc + pmh@rmi@pm)@pmh
        s22 = np.linalg.inv(qmc + pmh@rmi@pm)@(qmc - pmh@rmi@pm)
    

    dia = pm@rmi@pm.T
    smatrix_full = np.block([[s11, s12],[s21, s22]])
    
    return smatrix_full 
            
def plot_mode(mode = 'TE', f = 100e9, n = 1,m = 1, a = 1e-3, polar = 0):
    """ 
    Visualize the mode 

    Arguments
    ---------
    mode - str ('TE' or 'TM') 
    f - float (in Hz. default: 100GHz). 
    n - int Azimuthal number 
    m - int Radial number 
    a - float radius of waveguide 
    polar - float. Polarization of the mode. 
    """
    if mode == 'TE':
        Efield, Hfield,_ = TE(n,m,a,polar)
    else:
        Efield, Hfield,_ = TM(n,m,a,polar)
        #Create the grid to compute over
    
    rad_list = np.linspace(0.0001, a, 10)
    theta_list = np.linspace(0,2*np.pi, 50)
    theta, r = np.meshgrid(theta_list, rad_list)
    #Create empty lists for E and H fields
    efield_r_matrix = np.zeros((10, 50))
    hfield_r_matrix = np.zeros((10, 50))
    efield_theta_matrix = np.zeros((10, 50))
    hfield_theta_matrix = np.zeros((10,50))
    efield_z_matrix  = np.zeros((10,50))
    hfield_z_matrix = np.zeros((10,50))
    #Loop over the mesh grid 
    index = 0
    for rad_index in rad_list:
        jndex = 0
        for theta_index in theta_list:
            #Compute the values
            efield_at_point = Efield(f, rad_index, theta_index)
            hfield_at_point = Hfield(f, rad_index, theta_index)
            
            #Store the values in the matrix
            efield_r_matrix[index, jndex] = np.imag(efield_at_point[0])
            efield_theta_matrix[index,jndex] = np.imag(efield_at_point[1])
            efield_z_matrix[index, jndex] = np.imag(efield_at_point[2])
            hfield_r_matrix[index, jndex] = np.imag(hfield_at_point[0])
            hfield_theta_matrix[index, jndex] = np.imag(hfield_at_point[1])
            hfield_z_matrix[index, jndex] = np.imag(hfield_at_point[2])
            
            jndex = jndex + 1
        index = index + 1
        
    #Create the plot
    ax = plt.subplot(111, polar = True)
    ax.quiver(theta, r, efield_r_matrix*np.cos(theta) - efield_theta_matrix*np.sin(theta), efield_r_matrix*np.sin(theta) + efield_theta_matrix*np.cos(theta))
    
    plt.title(mode + str(n) + str(m) + ' mode E field distribution')
    plt.show()


    ax = plt.subplot(111, polar = True)
    ax.quiver(theta, r, hfield_r_matrix*np.cos(theta) - hfield_theta_matrix*np.sin(theta), hfield_r_matrix*np.sin(theta) + hfield_theta_matrix*np.cos(theta))

    plt.title(mode + str(n) + str(m) + ' mode H field distribution')
    plt.show()

    '''
    ax = plt.subplot(111, polar = True)
    ax.contourf(theta, r, np.abs(efield_r_matrix**2 + efield_theta_matrix**2 + efield_z_matrix**2))
    plt.title(mode + str(n) +str(m) + ' mode E field ABS distribution ') 
    plt.show()
    '''                
        
    
    
