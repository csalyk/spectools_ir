import numpy as np
from astroquery.hitran import Hitran
from astropy import units as un
from astropy.constants import c, k_B, h, u
from astropy import units as un
from scipy.optimize import curve_fit
from spectools_ir.utils import fwhm_to_sigma, sigma_to_fwhm

def _convert_quantum_strings(hitran_data_in):
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
    if('gp' in hitran_data.columns): hitran_data.rename_column('gp','gup')
    if('gpp' in hitran_data.columns): hitran_data.rename_column('gpp','glow')
    if('Vp' in hitran_data.columns): hitran_data.rename_column('Vp','Vp_HITRAN')
    if('Vpp' in hitran_data.columns): hitran_data.rename_column('Vpp','Vpp_HITRAN')
    if('Qp' in hitran_data.columns): hitran_data.rename_column('Qp','Qp_HITRAN')
    if('Qpp' in hitran_data.columns): hitran_data.rename_column('Qpp','Qpp_HITRAN')
    if('Vp' in hitran_data.columns): hitran_data['Vup']=np.zeros(nlines)
    if('Vpp' in hitran_data.columns): hitran_data['Vlow']=np.zeros(nlines)
    if('Qp' in hitran_data.columns): hitran_data['Qup']=np.zeros(nlines)
    if('Qpp' in hitran_data.columns): hitran_data['Qlow']=np.zeros(nlines)
    if(('Vp_HITRAN' in hitran_data.columns) and ('Vup' in hitran_data.columns) and ('Vlow' in hitran_data.columns) and ('Qpp_HITRAN' in hitran_data.columns) and ('molec_id' in hitran_data.columns) ):  
        for i,myvp in enumerate(hitran_data['Vp_HITRAN']):
            if(hitran_data['molec_id'][i]==5):   #Special formatting specific to rovibrational CO
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


def _strip_superfluous_hitran_data(hitran_data_in):
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
    if('sw' in hitran_data.columns): del hitran_data['sw']
    if('gamma_air' in hitran_data.columns): del hitran_data['gamma_air']
    if('gamma_self' in hitran_data.columns): del hitran_data['gamma_self']
    if('n_air' in hitran_data.columns): del hitran_data['n_air']
    if('delta_air' in hitran_data.columns): del hitran_data['delta_air']
    if('ierr1' in hitran_data.columns): del hitran_data['ierr1']
    if('ierr2' in hitran_data.columns): del hitran_data['ierr2']
    if('ierr3' in hitran_data.columns): del hitran_data['ierr3']
    if('ierr4' in hitran_data.columns): del hitran_data['ierr4']
    if('ierr5' in hitran_data.columns): del hitran_data['ierr5']
    if('ierr6' in hitran_data.columns): del hitran_data['ierr6']
    if('iref1' in hitran_data.columns): del hitran_data['iref1']
    if('iref2' in hitran_data.columns): del hitran_data['iref2']
    if('iref3' in hitran_data.columns): del hitran_data['iref3']
    if('iref4' in hitran_data.columns): del hitran_data['iref4']
    if('iref5' in hitran_data.columns): del hitran_data['iref5']
    if('iref6' in hitran_data.columns): del hitran_data['iref6']
    if('line_mixing_flag' in hitran_data.columns): del hitran_data['line_mixing_flag']
    return hitran_data        
