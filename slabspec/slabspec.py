import numpy as np
import urllib
import pandas as pd

from astropy.table import Table
from astropy import units as un
from astropy.io import fits
from astropy.constants import c,h, k_B, G, M_sun, au, pc, u
from astropy.convolution import Gaussian1DKernel, convolve

from spectools_ir.utils import fwhm_to_sigma, sigma_to_fwhm, compute_thermal_velocity, extract_hitran_data
from spectools_ir.utils import  get_molecule_identifier, get_global_identifier, spec_convol
from .helpers import _strip_superfluous_hitran_data, _convert_quantum_strings

def _spec_convol_klaus(wave,flux,R):
    '''
    Convolve a spectrum, given wavelength in microns and flux density, by a given FWHM in velocity 

    Parameters
    ---------
    wave : numpy array
        wavelength values, in microns
    flux : numpy array
        flux density values, in units of Energy/area/time/Hz
    R : float
        Resolving power (lambda / dlambda)

    Returns
    --------
    newflux : numpy array
        Convolved spectrum flux density values, in same units as input

    '''                                 
    # find the minimum spacing between wavelengths in the dataset
    dws = np.abs(wave - np.roll(wave, 1))
    dw_min = np.min(dws)   #Minimum delta-wavelength between points in dataset

    fwhm = wave / R  # FWHM of resolution element as a function of wavelength ("delta lambda" in same units as wave)
    #fwhm / dw_min gives FWHM values expressed in units of minimum spacing, or the sampling for each wavelength
    #(sampling is sort of the number of data points per FWHM)
    #The sampling is different for each point in the wavelength array, because the FWHM is wavelength dependent
    #fwhm_s then gives the minimum value of the sampling - the most poorly sampled wavelength.
    fwhm_s = np.min(fwhm / dw_min)  # find mininumvalue of sampling for this dataset
    # but do not allow the sampling FWHM to be less than Nyquist
    # (i.e., make sure there are at least two points per resolution element)
    fwhm_s = np.max([2., fwhm_s])  #Will return 2 only if fwhm_s is less than 2
    #If you want all wavelengths to have the same sampling per resolution element, 
    #then this ds gives the wavelength spacing for each wavelength (in units of wavelength)
    ds = fwhm / fwhm_s
    # use the min wavelength as a starting point
    w = np.min(wave)
    #Initialize array to hold new wavelength values
    #Note: it's much faster (~50%) to append to lists than np.array()'s
    wave_constfwhm = []


    # doing this as a loop is slow, but straightforward.
    while w < np.max(wave):
        # use interpolation to get delta-wavelength from the sampling as a function of wavelength.
        # this method is over 5x faster than the original use of scipy.interpolate.interp1d.
        w += np.interp(w,wave,ds)  #Get value of ds at w, then add to old value of w
        wave_constfwhm.append(w)

    wave_constfwhm.pop()  # remove last point which is an extrapolation
    wave_constfwhm = np.array(wave_constfwhm)  #Convert list to numpy array
    
    # interpolate the flux onto the new wavelength set
    flux_constfwhm = np.interp(wave_constfwhm,wave,flux)

    # convolve the flux with a gaussian kernel; first convert the FWHM to sigma
    sigma_s = fwhm_s / 2.3548
    try:
        # for astropy < 0.4
        g = Gaussian1DKernel(width=sigma_s)
    except TypeError:
        # for astropy >= 0.4
        g = Gaussian1DKernel(sigma_s)
    # use boundary='extend' to set values outside the array to nearest array value.
    # this is the best approximation in this case.
    flux_conv = convolve(flux_constfwhm, g, normalize_kernel=True, boundary='extend')
    flux_oldsampling = np.interp(wave, wave_constfwhm, flux_conv)

    return flux_oldsampling

#------------------------------------------------------------------------------------                                     
def make_spec(molecule_name, n_col, temp, area, wmax=40, wmin=1, res=1e-4, deltav=None, isotopologue_number=1, d_pc=1,
              aupmin=None, convol_fwhm=None, eupmax=None, vup=None, swmin=None):

    '''
    Create an IR spectrum for a slab model with given temperature, area, and column density

    Parameters
    ---------
    molecule_name : string
        String identifier for molecule, for example, 'CO', or 'H2O'             
    n_col : float
        Column density, in m^-2
    temp : float
        Temperature of slab model, in K
    area : float
        Area of slab model, in m^2
    wmin : float, optional
        Minimum wavelength of output spectrum, in microns. Defaults to 1 micron.
    wmax : float, optional
        Maximum wavelength of output spectrum, in microns.  Defaults to 40 microns.
    deltav : float, optional
        sigma of local velocity distribution, in m/s.  Note this is NOT the global velocity distribution.
        Defaults to thermal speed of molecule given input temperature.
    isotopologue_number : float, optional
        Number representing isotopologue (1=most common, 2=next most common, etc.)
    d_pc : float, optional
        Distance to slab, in units of pc, for computing observed flux density.  Defaults to 1 pc.
    aupmin : float, optional
        Minimum Einstein-A coefficient for transitions
    swmin : float, optional
        Minimum line strength for transitions
    convol_fwhm : float, optional
        FWHM of convolution kernel, in km/s.
    res : float, optional
        max resolution of spectrum, in microns.  Must be significantly higher than observed spectrum for correct calculation.
        Defaults to 1e-4.
    eupmax : float, optional
        Maximum energy of transitions to consider, in K
    vup : float, optional
        Optional parameter to restrict output to certain upper level vibrational states.  Only works if 'Vp' field is a single integer.

    Returns
    --------
    slabdict : dictionary
        Dictionary includes two astropy tables: 
          lineparams : line parameters from HITRAN, integrated line fluxes, peak tau
          spectrum : wavelength, flux, convolflux, tau
        and two dictionaries
          lines : wave_arr (in microns), flux_arr (in mks), velocity (in km/s) - for plotting individual lines
          modelparams : model parameters: Area, column density, temperature, local velocity, convolution fwhm
    '''
    isot=isotopologue_number
    si2jy=1e26   #SI to Jy flux conversion factor

#If local velocity field is not given, assume sigma given by thermal velocity
    if(deltav is None):
        deltav=compute_thermal_velocity(molecule_name, temp)

#Read HITRAN data
    hitran_data=extract_hitran_data(molecule_name,wmin,wmax,isotopologue_number=isotopologue_number, eupmax=eupmax, aupmin=aupmin, swmin=swmin, vup=vup)

    wn0=hitran_data['wn']*1e2 # now m-1
    aup=hitran_data['a']
    eup=(hitran_data['elower']+hitran_data['wn'])*1e2 #now m-1                                                             
    gup=hitran_data['gp']

#Compute partition function
    q=_compute_partition_function(molecule_name,temp,isot)
    
#Begin calculations                                                                                                       
    afactor=((aup*gup*n_col)/(q*8.*np.pi*(wn0)**3.)) #mks                                                                 
    efactor=h.value*c.value*eup/(k_B.value*temp)
    wnfactor=h.value*c.value*wn0/(k_B.value*temp)
    phia=1./(deltav*np.sqrt(2.0*np.pi))
    efactor2=hitran_data['eup_k']/temp
    efactor1=hitran_data['elower']*1.e2*h.value*c.value/k_B.value/temp
    tau0=afactor*(np.exp(-1.*efactor1)-np.exp(-1.*efactor2))*phia  #Avoids numerical issues at low T
    w0=1.e6/wn0

    dvel=0.1e0    #km/s
    nvel=1001
    vel=(dvel*(np.arange(0,nvel)-500.0))*1.e3     #now in m/s   

    omega=area/(d_pc*pc.value)**2.
    fthin=aup*gup*n_col*h.value*c.value*wn0/(q*4.*np.pi)*np.exp(-efactor)*omega # Energy/area/time, mks                   

#Now loop over transitions and velocities to calculate flux                                                               
    nlines=np.size(tau0)
    tau=np.zeros([nlines,nvel])
    wave=np.zeros([nlines,nvel])
    for ha,mytau in enumerate(tau0):
        for ka, myvel in enumerate(vel):
            tau[ha,ka]=tau0[ha]*np.exp(-vel[ka]**2./(2.*deltav**2.))
            wave[ha,ka]=1.e6/wn0[ha]*(1+vel[ka]/c.value)
#Now interpolate over wavelength space so that all lines can be added together                                            
    w_arr=wave            #nlines x nvel                                                                                  
    f_arr=w_arr-w_arr     #nlines x nvel                                                                                  
    nbins=(wmax-wmin)/res
#Create arrays to hold full spectrum (optical depth vs. wavelength)
    totalwave=np.arange(nbins)*(wmax-wmin)/nbins+wmin
    totaltau=np.zeros(np.size(totalwave))

#Create array to hold line fluxes (one flux value per line)
    lineflux=np.zeros(nlines)
    for i in range(nlines):
        w=np.where((totalwave > np.min(wave[i,:])) & (totalwave < np.max(wave[i,:])))
        if(np.size(w) > 0):
            newtau=np.interp(totalwave[w],wave[i,:], tau[i,:])
            totaltau[w]+=newtau
            f_arr[i,:]=2*h.value*c.value*wn0[i]**3./(np.exp(wnfactor[i])-1.0e0)*(1-np.exp(-tau[i,:]))*si2jy*omega
            lineflux_jykms=np.sum(f_arr[i,:])*dvel
            lineflux[i]=lineflux_jykms*1e-26*1.*1e5*(1./(w0[i]*1e-4))    #mks

    wave_arr=wave
    wn=1.e6/totalwave                                         #m^{-1}                                                     
    wnfactor=h.value*c.value*wn/(k_B.value*temp)
    flux=2*h.value*c.value*wn**3./(np.exp(wnfactor)-1.0e0)*(1-np.exp(-totaltau))*si2jy*omega

    wave=totalwave
    #convol_fwhm should be set to FWHM of convolution kernel, in km/s                                                          
    convolflux=np.copy(flux)
    if(convol_fwhm is not None):
        convolflux=spec_convol(wave,flux,convol_fwhm)

    slabdict={}

#Line params
    hitran_data['lineflux']=lineflux
    hitran_data['tau_peak']=tau0
    hitran_data['fthin']=fthin
    hitran_data=_convert_quantum_strings(hitran_data)
    hitran_data=_strip_superfluous_hitran_data(hitran_data)
    slabdict['lineparams']=hitran_data

#Line flux array
    lines={'flux_arr':f_arr , 'wave_arr':wave_arr , 'velocity':vel*1e-3}
    slabdict['lines']=lines

#Spectrum
    spectrum_table = Table([wave, flux, convolflux, totaltau], names=('wave', 'flux', 'convolflux','totaltau'),  dtype=('f8', 'f8', 'f8','f8'))
    spectrum_table['wave'].unit = 'micron'
    spectrum_table['flux'].unit = 'Jy'
    spectrum_table['convolflux'].unit = 'Jy'
    slabdict['spectrum']=spectrum_table

#Model params
    if(convol_fwhm is not None):
        convol_fwhm=convol_fwhm*un.km/un.s
    modelparams_table={'area':area*un.meter*un.meter,'temp':temp*un.K,'n_col':n_col/un.meter/un.meter, 'res':res*un.micron, 
                       'deltav':deltav*un.meter/un.s, 'convol_fwhm':convol_fwhm, 'd_pc':d_pc*un.parsec,
                       'isotopologue_number':isot,'molecule_name':molecule_name}
    slabdict['modelparams']=modelparams_table

    return slabdict


def _compute_partition_function(molecule_name,temp,isotopologue_number=1):
    '''                                                                                                                                       
    For a given input molecule name, isotope number, and temperature, return the partition function Q
                                                                                                                                              
    Parameters                                                                                                                                
    ----------                                                                                                                                
    molecule_name : string
        The molecule name string (e.g., 'CO', 'H2O')
    temp : float
        The temperature at which to compute the partition function
    isotopologue_number : float, optional
        Isotopologue number, with 1 being most common, etc. Defaults to 1.

    Returns                                                                                                                                   
    -------                                                                                                                                   
    q : float
      The partition function
    '''

    G=get_global_identifier(molecule_name, isotopologue_number=isotopologue_number)
    qurl='https://hitran.org/data/Q/'+'q'+str(G)+'.txt'
    handle = urllib.request.urlopen(qurl)
    qdata = pd.read_csv(handle,sep=' ',skipinitialspace=True,names=['temp','q'],header=None)

#    pathmod=os.path.dirname(__file__)
#    if not os.path.exists(qfilename):  #download data from internet
       #get https://hitran.org/data/Q/qstr(G).txt

    q=np.interp(temp,qdata['temp'],qdata['q'])
    return q

