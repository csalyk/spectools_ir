import sys

import numpy as np
import urllib
import pandas as pd
import pdb as pdb

from scipy.interpolate import interp1d

from astropy.table import Table
from astropy import units as un
from astropy.io import fits
from astropy.constants import c,h, k_B, G, M_sun, au, pc, u
from astropy.convolution import Gaussian1DKernel, convolve_fft

from spectools_ir.utils import _check_hitran
from spectools_ir.utils import fwhm_to_sigma, sigma_to_fwhm, compute_thermal_velocity, extract_hitran_data
from spectools_ir.utils import get_molecule_identifier, get_global_identifier, spec_convol, extract_hitran_from_par
from .helpers import _strip_superfluous_hitran_data, _convert_quantum_strings

#------------------------------------------------------------------------------------
def make_spec(molecule_name, n_col, temp, area, wmax=40, wmin=1, deltav=None, isotopologue_number=1, d_pc=1,
              aupmin=None, convol_fwhm=None, eupmax=None, vup=None, swmin=None, parfile=None, hitran_data=None):

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
    eupmax : float, optional
        Maximum energy of transitions to consider, in K
    vup : float, optional
        Optional parameter to restrict output to certain upper level vibrational states.  Only works if 'Vp' field is a single integer.
    parfile : string
        HITRAN ".par" file, used in place of the HITRAN API if provided
    hitran_data : astropy table
        An astropy table in the same format as output from extract_hitran_data.  Will be used in place of HITRAN API call if provided.
  

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
    oversamp = 3
    isot = isotopologue_number
    si2jy = 1e26   #SI to Jy flux conversion factor

    #Test whether molecule is in HITRAN database.  If not, check for parfile and warn.
    database = _check_hitran(molecule_name)
    if((database=='exomol') & (parfile is None)):
        print('This molecule is not in the HITRAN database.  You must provide a HITRAN-format parfile for this molecule.  Exiting.')
        sys.exit()
    if(database is None):
        print('This molecule is not covered by this code at this time.  Exiting.')
        sys.exit()

    #If local velocity field is not given, assume sigma given by thermal velocity
    if(deltav is None):
        deltav = compute_thermal_velocity(molecule_name, temp)

    #Read HITRAN data, unless already provided
    if(parfile is not None):
        hitran_data = extract_hitran_from_par(parfile,aupmin=aupmin,eupmax=eupmax,isotopologue_number=isotopologue_number,vup=vup,wavemin=wmin,wavemax=wmax)
    elif(hitran_data is None):  #parfile not provided, and hitran_data not provided.  Read using extract_hitran_data
        try:
            hitran_data = extract_hitran_data(molecule_name,wmin,wmax,isotopologue_number=isotopologue_number, eupmax=eupmax, aupmin=aupmin, swmin=swmin, vup=vup)
        except:
            print("astroquery call to HITRAN failed. This can happen when your molecule does not have any lines in the requested wavelength region")
            sys.exit(1)

    wn0 = hitran_data['wn']*1e2 # now m-1
    aup = hitran_data['a']
    eup = (hitran_data['elower']+hitran_data['wn'])*1e2 #now m-1
    if(not ('gp' in hitran_data.columns)): #Assuming gupper is either labeled gup or gp
        gup=hitran_data['gup']
    else:
        gup = hitran_data['gp']

    #Compute partition function
    q = _compute_partition_function(molecule_name,temp,isot)

    #Begin calculations
    afactor = ((aup*gup*n_col)/(q*8.*np.pi*(wn0)**3.)) #mks
    efactor = h.value*c.value*eup/(k_B.value*temp)
    wnfactor = h.value*c.value*wn0/(k_B.value*temp)
    phia = 1./(deltav*np.sqrt(2.0*np.pi))
    efactor2 = hitran_data['eup_k']/temp
    efactor1 = hitran_data['elower']*1.e2*h.value*c.value/k_B.value/temp
    tau0 = afactor*(np.exp(-1.*efactor1)-np.exp(-1.*efactor2))*phia  #Avoids numerical issues at low T

    dvel = deltav/oversamp    #m/s
    nvel = 10*oversamp+1 #5 sigma window
    vel = (dvel*(np.arange(0,nvel)-(nvel-1)/2))

    omega = area/(d_pc*pc.value)**2.
    fthin = aup*gup*n_col*h.value*c.value*wn0/(q*4.*np.pi)*np.exp(-efactor)*omega # Energy/area/time, mks

    #Now loop over transitions and velocities to calculate flux
    nlines = np.size(tau0)
    tau = np.zeros([nlines,nvel])
    wave = np.zeros([nlines,nvel])
    for ha,mytau in enumerate(tau0):
        tau[ha,:] = tau0[ha]*np.exp(-vel**2./(2.*deltav**2.))
        wave[ha,:] = 1.e6/wn0[ha]*(1+vel/c.value)

    #Now interpolate over wavelength space so that all lines can be added together
    w_arr = wave            #nlines x nvel
    f_arr = w_arr-w_arr     #nlines x nvel
    nbins = int(oversamp*(wmax-wmin)/wmax*(c.value/deltav))

    #Create arrays to hold full spectrum (optical depth vs. wavelength)
    totalwave = np.logspace(np.log10(wmin-10*deltav/c.value*wmax),np.log10(wmax+10*deltav/c.value*wmax),nbins) #Extend beyond input wave by 10xdelta_wave
    totaltau = np.zeros(nbins)

    #Create array to hold line fluxes (one flux value per line)
    lineflux = np.zeros(nlines)
    totalwave_index = np.arange(totalwave.size)
    index_interp = interp1d(totalwave,totalwave_index)
    for i in range(nlines):

        minw = np.min(wave[i,:])
        maxw = np.max(wave[i,:])
        minindex = int(index_interp(minw))
        maxindex = int(index_interp(maxw))

        w = np.arange(minindex,maxindex)

        if(np.size(w) > 0):
            newtau = np.interp(totalwave[w],wave[i,:], tau[i,:])
            totaltau[w] += newtau
            f_arr[i,:] = 2*h.value*c.value*wn0[i]**3./(np.exp(wnfactor[i])-1.0e0)*(1-np.exp(-tau[i,:]))*omega
            lineflux[i] = np.sum(f_arr[i,:]) * (dvel/c.value) * (c.value*wn0[i]) #in W/m2

    wave_arr = wave
    wn = 1.e6/totalwave                                         #m^{-1}
    wnfactor = h.value*c.value*wn/(k_B.value*temp)
    flux = 2*h.value*c.value*wn**3./(np.exp(wnfactor)-1.0e0)*(1-np.exp(-totaltau))*si2jy*omega

    wave = totalwave

    #convol_fwhm should be set to FWHM of convolution kernel, in km/s
    convolflux = np.copy(flux)
    if(convol_fwhm is not None):
        convolflux = spec_convol(wave,flux,convol_fwhm)

    slabdict={}

    #Line params
    hitran_data['lineflux'] = lineflux
    hitran_data['tau_peak'] = tau0
    hitran_data['fthin'] = fthin
    hitran_data = _convert_quantum_strings(hitran_data)
    hitran_data = _strip_superfluous_hitran_data(hitran_data)
    slabdict['lineparams'] = hitran_data

    #Line flux array
    lines={'flux_arr':f_arr , 'wave_arr':wave_arr , 'velocity':vel*1e-3}
    slabdict['lines'] = lines

    #Spectrum
    spectrum_table = Table([wave, flux, convolflux, totaltau], names=('wave', 'flux', 'convolflux','totaltau'),  dtype=('f8', 'f8', 'f8','f8'))
    spectrum_table['wave'].unit = 'micron'
    spectrum_table['flux'].unit = 'Jy'
    spectrum_table['convolflux'].unit = 'Jy'
    slabdict['spectrum']=spectrum_table

    #Model params
    if(convol_fwhm is not None):
        convol_fwhm=convol_fwhm*un.km/un.s
    modelparams_table={'area':area*un.meter*un.meter,'temp':temp*un.K,'n_col':n_col/un.meter/un.meter, 
                       'deltav':deltav*un.meter/un.s, 'convol_fwhm':convol_fwhm, 'd_pc':d_pc*un.parsec,
                       'isotopologue_number':isot,'molecule_name':molecule_name}
    slabdict['modelparams'] = modelparams_table

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

    exomol_pf_dict = {200:'https://www.exomol.com/db/SiO/28Si-16O/SiOUVenIR/28Si-16O__SiOUVenIR.pf',
                      201:'https://www.exomol.com/db/H2/1H2/RACPPK/1H2__RACPPK.pf'}

    G = get_global_identifier(molecule_name, isotopologue_number=isotopologue_number)
    if(G<200):  #in HITRAN database
        qurl = 'https://hitran.org/data/Q/'+'q'+str(G)+'.txt'
    if(G>=200):  #presumed to be in exomol
        qurl = exomol_pf_dict[G]

    handle = urllib.request.urlopen(qurl)
    print('Reading partition function from: ',qurl)
    qdata = pd.read_csv(handle,sep=' ',skipinitialspace=True,names=['temp','q'],header=None)

#    pathmod=os.path.dirname(__file__)
#    if not os.path.exists(qfilename):  #download data from internet
       #get https://hitran.org/data/Q/qstr(G).txt

    q = np.interp(temp,qdata['temp'],qdata['q'])
    return q

def write_slab(slabdict,filename='slabmodel.fits'):
    '''
    Write the slab model to a fits file.

    Parameters
    ----------
    slabdict : Dictionary
        Dictionary as output by slabspec
    filename : String
        Name of output fits file

    Returns
    -------
    '''

    wave = slabdict['spectrum']['wave']
    flux = slabdict['spectrum']['convolflux']

    c1 = fits.Column(name='wave', array=wave, format='F')
    c2 = fits.Column(name='flux', array=flux, format='F')
    t1 = fits.BinTableHDU.from_columns([c1, c2])

    moldata = slabdict['lineparams']
    mol_cols = []
    for key in moldata.keys():
        try:
            mol_cols.append(fits.Column(name=key,array=moldata[key],format='F'))
        except:
            mol_cols.append(fits.Column(name=key,array=moldata[key],format='A'))
    t2 = fits.BinTableHDU.from_columns(mol_cols)

    primary = fits.PrimaryHDU()
    hdulist = fits.HDUList([primary,t1,t2])

    hdulist.writeto(filename,overwrite=True)
