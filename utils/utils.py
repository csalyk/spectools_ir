import numpy as np
from astroquery.hitran import Hitran
from astropy import units as un
from astropy.constants import c, k_B, h, u
from molmass import Formula
import pdb as pdb

def make_rotation_diagram(lineparams, units='mks'):
    '''                                                                                                     
    Take ouput of make_spec and use it to compute rotation diagram parameters.                              
                                                                                                            
    Parameters                                                                                              
    ---------                                                                                               
    lineparams: dictionary                                                                                  
        dictionary output from make_spec                                                                    
                                                                                                            
    Returns                                                                                                 
    --------                                                                                                
    rot_table: astropy Table                                                                                
        Table of x and y values for rotation diagram.                                                       
                                                                                                            
    '''

    if('gup' in lineparams.columns):
        gup=lineparams['gup']

    if('gp' in lineparams.columns):
        gup=lineparams['gp']

    x=lineparams['eup_k']
    y=np.log(lineparams['lineflux']/(lineparams['wn']*1e2*gup*lineparams['a']))   #All mks
    if(units=='cgs'):
        y=np.log(1000.*lineparams['lineflux']/(lineparams['wn']*gup*lineparams['a'])) #All cgs
    if(units=='mixed'):
        y=np.log(lineparams['lineflux']/(lineparams['wn']*gup*lineparams['a'])) #Mixed units
    rot_dict={'x':x,'y':y,'units':units}

    return rot_dict


def compute_thermal_velocity(molecule_name, temp):
    '''
    Compute the thermal velocity given a molecule name and temperature

    Parameters
    ---------
    molecule_name: string
      Molecule name (e.g., 'CO', 'H2O')
    temp : float
      Temperature at which to compute thermal velocity

    Returns
    -------
    v_thermal : float
       Thermal velocity (m/s)
    '''

    f=Formula(molecule_name)
    mu=f.isotope.mass*u.value   #kg

    return np.sqrt(k_B.value*temp/mu)   #m/s

def markgauss(x,mean=0, sigma=1., area=1):
    '''
    Compute a Gaussian function

    Parameters
    ----------
    x : float
      x values at which to calculate the Gaussian
    mean : float, optional
      mean of Gaussian
    sigma : float, optional
      standard deviation of Gaussian
    area : float, optional
      area of Gaussian curve

    Returns
    ---------
    Gaussian function evaluated at input x values

    '''

    norm=area
    u = ( (x-mean)/np.abs(sigma) )**2
    norm = norm / (np.sqrt(2. * np.pi)*sigma)
    f=norm*np.exp(-0.5*u)

    return f

def sigma_to_fwhm(sigma):
    '''
    Convert sigma to fwhm

    Parameters
    ----------
    sigma : float
       sigma of Gaussian distribution

    Returns
    ----------
    fwhm : float
       Full Width at Half Maximum of Gaussian distribution
    '''                        
    return  sigma*(2.*np.sqrt(2.*np.log(2.)))

def fwhm_to_sigma(fwhm):
    '''
    Convert fwhm to sigma

    Parameters
    ----------
    fwhm : float
       Full Width at Half Maximum of Gaussian distribution

    Returns
    ----------
    sigma : float
       sigma of Gaussian distribution
    '''                        

    return fwhm/(2.*np.sqrt(2.*np.log(2.)))

def wn_to_k(wn):
    '''                        
    Convert wavenumber to Kelvin

    Parameters
    ----------
    wn : AstroPy quantity
       Wavenumber including units

    Returns
    ---------
    energy : AstroPy quantity
       Energy of photon with given wavenumber

    '''              
    return wn.to(1/un.m)*h*c/k_B

def extract_hitran_data(molecule_name, wavemin, wavemax, isotopologue_number=1, eupmax=None, aupmin=None,swmin=None,vup=None):
    '''                                                               
    Extract data from HITRAN 
    Primarily makes use of astroquery.hitran, with some added functionality specific to common IR spectral applications
    Parameters 
    ---------- 
    molecule_name : string
        String identifier for molecule, for example, 'CO', or 'H2O'
    wavemin: float
        Minimum wavelength of extracted lines (in microns)
    wavemax: float
        Maximum wavelength of extracted lines (in microns)                   
    isotopologue_number : float, optional
        Number representing isotopologue (1=most common, 2=next most common, etc.)
    eupmax : float, optional
        Maximum extracted upper level energy (in Kelvin)
    aupmin : float, optional
        Minimum extracted Einstein A coefficient
    swmin : float, optional
        Minimum extracted line strength
    Returns
    ------- 
    hitran_data : astropy table
        Extracted data
    '''

    #Convert molecule name to number
    M = get_molecule_identifier(molecule_name)

    #Convert inputs to astroquery formats
    min_wavenumber = 1.e4/wavemax
    max_wavenumber = 1.e4/wavemin

    #Extract hitran data using astroquery
    tbl = Hitran.query_lines(molecule_number=M,isotopologue_number=isotopologue_number,min_frequency=min_wavenumber / un.cm,max_frequency=max_wavenumber / un.cm)

    #Do some desired bookkeeping, and add some helpful columns
    tbl.rename_column('nu','wn')
    tbl['nu']=tbl['wn']*c.cgs.value   #Now actually frequency of transition
    tbl['eup_k']=(wn_to_k((tbl['wn']+tbl['elower'])/un.cm)).value

    tbl['wave']=1.e4/tbl['wn']       #Wavelength of transition, in microns
    tbl.rename_column('global_upper_quanta','Vp')
    tbl.rename_column('global_lower_quanta','Vpp')
    tbl.rename_column('local_upper_quanta','Qp')
    tbl.rename_column('local_lower_quanta','Qpp')

    #Extract desired portion of dataset
    ebool = np.full(np.size(tbl), True, dtype=bool)  #default to True
    abool = np.full(np.size(tbl), True, dtype=bool)  #default to True
    swbool = np.full(np.size(tbl), True, dtype=bool)  #default to True
    vupbool = np.full(np.size(tbl), True, dtype=bool)  #default to True                                                            
    #Upper level energy
    if(eupmax is not None):
        ebool = tbl['eup_k'] < eupmax
    #Upper level A coeff
    if(aupmin is not None):
        abool = tbl['a'] > aupmin
    #Line strength
    if(swmin is not None):
        swbool = tbl['sw'] > swmin
   #Vup
    if(vup is not None):
        vupval = [np.int(val) for val in tbl['Vp']]
        vupbool=(np.array(vupval)==vup)
   #Combine
    extractbool = (abool & ebool & swbool & vupbool)
    hitran_data=tbl[extractbool]

    #Return astropy table
    return hitran_data

def get_global_identifier(molecule_name,isotopologue_number=1):
    '''                                                                                                                                
    For a given input molecular formula, return the corresponding HITRAN *global* identifier number.
    For more info, see https://hitran.org/docs/iso-meta/ 
                                                                                                                                       
    Parameters                                                                                                                         
    ----------                                                                                                                         
    molecular_formula : str                                                                                                            
        The string describing the molecule.              
    isotopologue_number : int, optional
        The isotopologue number, from most to least common.                                                                              
                                                                                                                                       
    Returns                                                                                                                            
    -------                                                                                                                            
    G : int                                                                                                                            
        The HITRAN global identifier number.                                                                                        
    '''

    mol_isot_code=molecule_name+'_'+str(isotopologue_number)

    trans = { 'H2O_1':1, 'H2O_2':2, 'H2O_3':3, 'H2O_4':4, 'H2O_5':5, 'H2O_6':6, 'H2O_7':129,
               'CO2_1':7,'CO2_2':8,'CO2_3':9,'CO2_4':10,'CO2_5':11,'CO2_6':12,'CO2_7':13,'CO2_8':14,
               'CO2_9':121,'CO2_10':15,'CO2_11':120,'CO2_12':122,
               'O3_1':16,'O3_2':17,'O3_3':18,'O3_4':19,'O3_5':20,
               'N2O_1':21,'N2O_2':22,'N2O_3':23,'N2O_4':24,'N2O_5':25,
               'CO_1':26,'CO_2':27,'CO_3':28,'CO_4':29,'CO_5':30,'CO_6':31,
               'CH4_1':32,'CH4_2':33,'CH4_3':34,'CH4_4':35,
               'O2_1':36,'O2_2':37,'O2_3':38,
               'NO_1':39,'NO_2':40,'NO_3':41,
               'SO2_1':42,'SO2_2':43,
               'NO2_1':44,
               'NH3_1':45,'NH3_2':46,
               'HNO3_1':47,'HNO3_2':117,
               'OH_1':48,'OH_2':49,'OH_3':50,
               'HF_1':51,'HF_2':110,
               'HCl_1':52,'HCl_2':53,'HCl_3':107,'HCl_4':108,
               'HBr_1':54,'HBr_2':55,'HBr_3':111,'HBr_4':112,
               'HI_1':56,'HI_2':113,
               'ClO_1':57,'ClO_2':58,
               'OCS_1':59,'OCS_2':60,'OCS_3':61,'OCS_4':62,'OCS_5':63,
               'H2CO_1':64,'H2CO_2':65,'H2CO_3':66,
               'HOCl_1':67,'HOCl_2':68,
               'N2_1':69,'N2_2':118,
               'HCN_1':70,'HCN_2':71,'HCN_3':72,
               'CH3Cl_1':73,'CH3CL_2':74,
               'H2O2_1':75,
               'C2H2_1':76,'C2H2_2':77,'C2H2_3':105,
               'C2H6_1':78,'C2H6_2':106,
               'PH3_1':79,
               'COF2_1':80,'COF2_2':119,
               'SF6_1':126,
               'H2S_1':81,'H2S_2':82,'H2S_3':83,
               'HCOOH_1':84,
               'HO2_1':85,
               'O_1':86,
               'ClONO2_1':127,'ClONO2_2':128,
               'NO+_1':87,
               'HOBr_1':88,'HOBr_2':89,
               'C2H4_1':90,'C2H4_2':91,
               'CH3OH_1':92,
               'CH3Br_1':93,'CH3Br_2':94,
               'CH3CN_1':95,
               'CF4_1':96,
               'C4H2_1':116,
               'HC3N_1':109,
               'H2_1':103,'H2_2':115,
               'CS_1':97,'CS_2':98,'CS_3':99,'CS_4':100,
               'SO3_1':114,
               'C2N2_1':123,
               'COCl2_1':124,'COCl2_2':125}
 
    return trans[mol_isot_code]

#Code from Nathan Hagen
#https://github.com/nzhagen/hitran
def translate_molecule_identifier(M):
    '''                                                                                                            
    For a given input molecule identifier number, return the corresponding molecular formula.                      
                                                                                                                   
    Parameters                                                                                                     
    ----------                                                                                                     
    M : int                                                                                                        
        The HITRAN molecule identifier number.                                                                     
                                                                                                                   
    Returns                                                                                                        
    -------                                                                                                        
    molecular_formula : str                                                                                        
        The string describing the molecule.                                                                        
    '''

    trans = { '1':'H2O',    '2':'CO2',   '3':'O3',      '4':'N2O',   '5':'CO',    '6':'CH4',   '7':'O2',     '8':'NO',
              '9':'SO2',   '10':'NO2',  '11':'NH3',    '12':'HNO3', '13':'OH',   '14':'HF',   '15':'HCl',   '16':'HBr',
             '17':'HI',    '18':'ClO',  '19':'OCS',    '20':'H2CO', '21':'HOCl', '22':'N2',   '23':'HCN',   '24':'CH3Cl',
             '25':'H2O2',  '26':'C2H2', '27':'C2H6',   '28':'PH3',  '29':'COF2', '30':'SF6',  '31':'H2S',   '32':'HCOOH',
             '33':'HO2',   '34':'O',    '35':'ClONO2', '36':'NO+',  '37':'HOBr', '38':'C2H4', '39':'CH3OH', '40':'CH3Br',
             '41':'CH3CN', '42':'CF4',  '43':'C4H2',   '44':'HC3N', '45':'H2',   '46':'CS',   '47':'SO3'}
    return(trans[str(M)])

#Code from Nathan Hagen
#https://github.com/nzhagen/hitran
def get_molecule_identifier(molecule_name):
    '''                                                                                                                                
    For a given input molecular formula, return the corresponding HITRAN molecule identifier number.                                   
                                                                                                                                       
    Parameters                                                                                                                         
    ----------                                                                                                                         
    molecular_formula : str                                                                                                            
        The string describing the molecule.                                                                                            
                                                                                                                                       
    Returns                                                                                                                            
    -------                                                                                                                            
    M : int                                                                                                                            
        The HITRAN molecular identifier number.                                                                                        
    '''

    trans = { '1':'H2O',    '2':'CO2',   '3':'O3',      '4':'N2O',   '5':'CO',    '6':'CH4',   '7':'O2',     '8':'NO',
              '9':'SO2',   '10':'NO2',  '11':'NH3',    '12':'HNO3', '13':'OH',   '14':'HF',   '15':'HCl',   '16':'HBr',
             '17':'HI',    '18':'ClO',  '19':'OCS',    '20':'H2CO', '21':'HOCl', '22':'N2',   '23':'HCN',   '24':'CH3Cl',
             '25':'H2O2',  '26':'C2H2', '27':'C2H6',   '28':'PH3',  '29':'COF2', '30':'SF6',  '31':'H2S',   '32':'HCOOH',
             '33':'HO2',   '34':'O',    '35':'ClONO2', '36':'NO+',  '37':'HOBr', '38':'C2H4', '39':'CH3OH', '40':'CH3Br',
             '41':'CH3CN', '42':'CF4',  '43':'C4H2',   '44':'HC3N', '45':'H2',   '46':'CS',   '47':'SO3'}
    ## Invert the dictionary.                                                                                                          
    trans = {v:k for k,v in trans.items()}
    return(int(trans[molecule_name]))

def spec_convol(wave, flux, dv):
    '''                                                                                                             
    Convolve a spectrum, given wavelength in microns and flux density, by a given FWHM in velocity                  
                                                                                                                    
    Parameters                                                                                                      
    ---------                                                                                                       
    wave : numpy array                                                                                              
        wavelength values, in microns                                                                               
    flux : numpy array                                                                                              
        flux density values, in units of Energy/area/time/Hz                                                        
    dv : float                                                                                                      
        FWHM of convolution kernel, in km/s                                                                         
                                                                                                                    
    Returns                                                                                                         
    --------                                                                                                        
    newflux : numpy array                                                                                           
        Convolved spectrum flux density values, in same units as input                                              
                                                                                                                    
    '''

#Program assumes units of dv are km/s, and dv=FWHM                                                                 \
                                                                                                                    

    dv=fwhm_to_sigma(dv)
    n=round(4.*dv/(c.value*1e-3)*np.median(wave)/(wave[1]-wave[0]))
    if (n < 10):
        n=10.

#Pad arrays to deal with edges                                                                                     \

    dwave=wave[1]-wave[0]
    wave_low=np.arange(wave[0]-dwave*n, wave[0]-dwave, dwave)
    wave_high=np.arange(np.max(wave)+dwave, np.max(wave)+dwave*(n-1.), dwave)
    nlow=np.size(wave_low)
    nhigh=np.size(wave_high)
    flux_low=np.zeros(nlow)
    flux_high=np.zeros(nhigh)
    mask_low=np.zeros(nlow)
    mask_high=np.zeros(nhigh)
    mask_middle=np.ones(np.size(wave))
    wave=np.concatenate([wave_low, wave, wave_high])
    flux=np.concatenate([flux_low, flux, flux_high])
    mask=np.concatenate([mask_low, mask_middle, mask_high])

    newflux=np.copy(flux)

    if( n > (np.size(wave)-n)):
        print("Your wavelength range is too small for your kernel")
        print("Program will return an empty array")

    for i in np.arange(n, np.size(wave)-n+1):
        lwave=wave[np.int(i-n):np.int(i+n+1)]
        lflux=flux[np.int(i-n):np.int(i+n+1)]
        lvel=(lwave-wave[np.int(i)])/wave[np.int(i)]*c.value*1e-3
        nvel=(np.max(lvel)-np.min(lvel))/(dv*.2) +3
        vel=np.arange(nvel)
        vel=.2*dv*(vel-np.median(vel))
        kernel=markgauss(vel,mean=0,sigma=dv,area=1.)
        wkernel=np.interp(lvel,vel,kernel)   #numpy interp is almost factor of 2 faster than interp1d               
        wkernel=wkernel/np.nansum(wkernel)
        newflux[np.int(i)]=np.nansum(lflux*wkernel)/np.nansum(wkernel[np.isfinite(lflux)])
        #Note: denominator is necessary to correctly account for NaN'd regions                                     \
                                                                                                                    

#Remove NaN'd regions                                                                                              \
                                                                                                                    
    nanbool=np.invert(np.isfinite(flux))   #Places where flux is not finite                                        \
                                                                                                                    
    newflux[nanbool]='NaN'

#Now remove padding                                                                                                \
                                                                                                                    
    newflux=newflux[mask==1]

    return newflux
