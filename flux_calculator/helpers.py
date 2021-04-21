import numpy as np
from astroquery.hitran import Hitran
from astropy import units as un
from astropy.constants import c, k_B, h, u
from astropy import units as un
from scipy.optimize import curve_fit
import pdb as pdb

def convert_quantum_strings(hitran_data_in):
    '''
    Converts Vp, Vpp, Qp and Qpp quantum number strings to more useful format for analysis.
    Takes HITRAN values and saves them to new fields, e.g., 'Vp_HITRAN'
   
    Parameters
    ------------
    hitran_data : astropy table
    astropy table containing HITRAN data

    molecule_name : string
    Moleule name, e.g., 'CO'

    Returns
    ----------
    hitran_data : astropy table
    astropy table containing converted quantum number fields
    '''
    hitran_data=hitran_data_in.copy()
    nlines=np.size(hitran_data)
    hitran_data.rename_column('gp','gup')
    hitran_data.rename_column('gpp','glow')
    hitran_data.rename_column('Vp','Vp_HITRAN')
    hitran_data.rename_column('Vpp','Vpp_HITRAN')
    hitran_data.rename_column('Qp','Qp_HITRAN')
    hitran_data.rename_column('Qpp','Qpp_HITRAN')
    hitran_data['Vup']=np.zeros(nlines)
    hitran_data['Vlow']=np.zeros(nlines)
    hitran_data['Qup']=np.zeros(nlines)
    hitran_data['Qlow']=np.zeros(nlines)
    for i,myvp in enumerate(hitran_data['Vp_HITRAN']):
        if(hitran_data['molec_id'][i]==5):
            hitran_data['Vup'][i]=np.int(myvp)  #Upper level vibrational state
            hitran_data['Vlow'][i]=np.int(hitran_data['Vpp_HITRAN'][i])   #Lower level vibrational state
            type=(hitran_data['Qpp_HITRAN'][i].split())[0]   #Returns P or R  
            num=np.int((hitran_data['Qpp_HITRAN'][i].split())[1])
            hitran_data['Qlow'][i]=num  #Lower level Rotational state
            if(type=='P'): 
                hitran_data['Qup'][i]=num-1  #Upper level Rotational state for P branch
            if(type=='R'): 
                hitran_data['Qup'][i]=num+1  #Upper level Rotational state for R branch

    return hitran_data     


def strip_superfluous_hitran_data(hitran_data_in):
    '''
    Strips hitran_data astropy table of columns superfluous for IR astro spectroscopy

    Parameters
    ----------
    hitran_data : astropy table
    HITRAN data extracted by extract_hitran_data.  Contains all original columns from HITRAN.

    Returns    
    ----------
    hitran_data : astropy table
    HITRAN data stripped of some superfluous columns
    '''
    hitran_data=hitran_data_in.copy()
    del hitran_data['sw']
    del hitran_data['gamma_air']
    del hitran_data['gamma_self']
    del hitran_data['n_air']
    del hitran_data['delta_air']
    del hitran_data['ierr1']
    del hitran_data['ierr2']
    del hitran_data['ierr3']
    del hitran_data['ierr4']
    del hitran_data['ierr5']
    del hitran_data['ierr6']
    del hitran_data['iref1']
    del hitran_data['iref2']
    del hitran_data['iref3']
    del hitran_data['iref4']
    del hitran_data['iref5']
    del hitran_data['iref6']
    del hitran_data['line_mixing_flag']
    return hitran_data        


def calc_linewidth(p,perr=None):
    '''
    Given Gaussian fit to Flux vs. wavelength in microns, find line width in km/s
   
    Parameters
    ----------
    p : numpy array
    parameters from Gaussian fit
    
    Returns                                                        
    ---------                                                             
    linewidth : float
    linewidth in km/s (FWHM) 

    '''
    linewidth_err=0*un.km/un.s
    linewidth=sigma_to_fwhm(p[2]/p[1]*c.value*1e-3*un.km/un.s)
    if(perr is not None): linewidth_err=sigma_to_fwhm(perr[2]/p[1]*c.value*1e-3*un.km/un.s)

    return (linewidth, linewidth_err)

def gauss3(x, a0, a1, a2):
    z = (x - a1) / a2
    y = a0 * np.exp(-z**2 / 2.)
    return y

def gauss4(x, a0, a1, a2, a3):
    z = (x - a1) / a2
    y = a0 * np.exp(-z**2 / 2.) + a3
    return y

def gauss5(x, a0, a1, a2, a3, a4):
    z = (x - a1) / a2
    y = a0 * np.exp(-z**2 / 2.) + a3 + a4 * x
    return y

def line_fit(wave,flux,nterms=4,p0=None,bounds=None):
    '''
    Take wave and flux values and perform a Gaussian fit

    Parameters
    ----------
    wave : numpy array
      Wavelength values in units of microns.  Should be an isolated line.  
    flux : numpy array
      Flux density values (F_nu) in units of Jy.  Should be an isolated line.

    Returns
    ---------
    linefit['parameters','yfit','resid'] : dictionary
       Dictionary containing fit parameters,fit values, and residual
    '''
    options={5:gauss5, 4:gauss4, 3:gauss3}
    fit_func=options[nterms]
    try:
        if(bounds is not None):
            fitparameters, fitcovariance = curve_fit(fit_func, wave, flux, p0=p0,bounds=bounds)
        else:
            fitparameters, fitcovariance = curve_fit(fit_func, wave, flux, p0=p0)
    except RuntimeError:
        print("Error - curve_fit failed")
        return -1
    perr = np.sqrt(np.diag(fitcovariance))
    fitoutput={"yfit":fit_func(wave,*fitparameters),"parameters":fitparameters,
               "covariance":fitcovariance,"resid":flux-fit_func(wave,*fitparameters),
               "parameter_errors":perr}
    return fitoutput

def extract_vup(hitran_data,vup_value):

    vbool=[]
    for myvp in hitran_data['Vp']:
        vbool.append(np.int(myvp)==vup_value)  
    out=hitran_data[vbool]
    return out

def calc_numerical_flux(myx,myy,pfit, sigflux=None):
    '''
    Take parameters from line fit and compute a line flux and error

    Parameters
    ----------
    pfit : list
      fit parameters from Gaussian fit (pfit must have 3-5 elements)
    sigflux : float
      rms residual of fit, for computing error on line flux.

    Returns
    ---------
    (line flux, lineflux_err) : tuple of astropy quantities
       The line flux and the line flux error
    '''

    if(np.size(pfit)==5): [a0,a1,a2,a3,a4]=pfit
    if(np.size(pfit)==4): [a0,a1,a2,a3]=pfit
    if(np.size(pfit)==3): 
        [a0,a1,a2]=pfit
        a3=0

    nufit=c.value/(a1*1e-6)  #Frequency of line, in s-1
    myflux=myy-a3
    myvel=(myx-a1)/a1*c.value
    maxvel=(3*np.abs(a2))/a1*c.value
    mybool=(np.abs(myvel)<maxvel)  #3 sigma on either side of line
    dwave=np.nanmean(np.diff(myx[mybool]))
    lineflux=np.nansum(myflux[mybool]*dwave)*1e-26*1e-6*nufit**2./c.value*un.W/un.m/un.m
    return lineflux

def calc_line_flux_from_fit(pfit, sigflux=None):
    '''
    Take parameters from line fit and compute a line flux and error

    Parameters
    ----------
    pfit : list
      fit parameters from Gaussian fit (pfit must have 3-5 elements)
    sigflux : float
      rms residual of fit, for computing error on line flux.

    Returns
    ---------
    (line flux, lineflux_err) : tuple of astropy quantities
       The line flux and the line flux error
    '''

    if(np.size(pfit)==5): [a0,a1,a2,a3,a4]=pfit
    if(np.size(pfit)==4): [a0,a1,a2,a3]=pfit
    if(np.size(pfit)==3): [a0,a1,a2]=pfit

#Add error catching for size of p<3 or >5

    nufit=c.value/(a1*1e-6)  #Frequency of line, in s-1
    lineflux=np.abs(a0)*1.e-26*np.sqrt(2*np.pi)*(np.abs(a2)*1.e-6*nufit**2./c.value)*un.W/un.m/un.m
    if(sigflux is not None):
        lineflux_err=1.e-26*np.sqrt(2.*np.pi)*1.e-6*nufit**2./c.value*np.abs(a2)*sigflux*un.W/un.m/un.m
    else:
        lineflux_err=None
    return (lineflux,lineflux_err)
