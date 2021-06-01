import numpy as np
from astroquery.hitran import Hitran
from astropy import units as un
from astropy.constants import c, k_B, h, u

def calc_solid_angle(radius,distance):
    '''
    Convenience function to calculate solid angle from radius and distance, assuming a disk shape.

    Parameters
    ----------
    radius : float
     radius value in AU
    distance : float
     distance value in parsec

    Returns
    ----------
    solid angle : float
      solid angle in steradians
    '''
    return np.pi*radius**2./(distance*206265.)**2.

def calc_radius(solid_angle,distance):
    '''
    Convenience function to calculate disk radius from solid angle and distance, assuming a disk shape.

    Parameters
    ----------
    solid_angle : float
     solid angle value in radians
    distance : float
     distance value in parsec

    Returns
    ----------
    radius : float
     disk radius in AU
    '''
    return (distance*206265)*np.sqrt(solid_angle/np.pi)

def get_molmass(molecule_name,isotopologue_number=1):
    '''                                                                                                                                
    For a given input molecular formula, return the corresponding molecular mass, in amu
                                                                                                                                       
    Parameters                                                                                                                         
    ----------                                                                                                                         
    molecular_formula : str                                                                                                            
        The string describing the molecule.              
    isotopologue_number : int, optional
        The isotopologue number, from most to least common.                                                                              
                                                                                                                                       
    Returns                                                                                                                            
    -------                                                                                                                            
    mu : float                                                                                                                            
        Molecular mass in amu
    '''

    mol_isot_code=molecule_name+'_'+str(isotopologue_number)

#https://hitran.org/docs/iso-meta/

    mass = { 'H2O_1':18.010565, 'H2O_2':20.014811, 'H2O_3':19.01478, 'H2O_4':19.01674, 
               'H2O_5':21.020985, 'H2O_6':20.020956, 'H2O_7':20.022915,
               'CO2_1':43.98983,'CO2_2':44.993185,'CO2_3':45.994076,'CO2_4':44.994045,
               'CO2_5':46.997431,'CO2_6':45.9974,'CO2_7':47.998322,'CO2_8':46.998291,
               'CO2_9':45.998262,'CO2_10':49.001675,'CO2_11':48.001646,'CO2_12':47.0016182378,
               'O3_1':47.984745,'O3_2':49.988991,'O3_3':49.988991,'O3_4':48.98896,'O3_5':48.98896,
               'N2O_1':44.001062,'N2O_2':44.998096,'N2O_3':44.998096,'N2O_4':46.005308,'N2O_5':45.005278,
               'CO_1':27.994915,'CO_2':28.99827,'CO_3':29.999161,'CO_4':28.99913,'CO_5':31.002516,'CO_6':30.002485,
               'CH4_1':16.0313,'CH4_2':17.034655,'CH4_3':17.037475,'CH4_4':18.04083,
               'O2_1':31.98983,'O2_2':33.994076,'O2_3':32.994045,
               'NO_1':29.997989,'NO_2':30.995023,'NO_3':32.002234,
               'SO2_1':63.961901,'SO2_2':65.957695,
               'NO2_1':45.992904,'NO2_2':46.989938,
               'NH3_1':17.026549,'NH3_2':18.023583,
               'HNO3_1':62.995644,'HNO3_2':63.99268,
               'OH_1':17.00274,'OH_2':19.006986,'OH_3':18.008915,
               'HF_1':20.006229,'HF_2':21.012404,
               'HCl_1':35.976678,'HCl_2':37.973729,'HCl_3':36.982853,'HCl_4':38.979904,
               'HBr_1':79.92616,'HBr_2':81.924115,'HBr_3':80.932336,'HBr_4':82.930289,
               'HI_1':127.912297,'HI_2':128.918472,
               'ClO_1':50.963768,'ClO_2':52.960819,
               'OCS_1':59.966986,'OCS_2':61.96278,'OCS_3':60.970341,'OCS_4':60.966371,'OCS_5':61.971231, 'OCS_6':62.966136,
               'H2CO_1':30.010565,'H2CO_2':31.01392,'H2CO_3':32.014811,
               'HOCl_1':51.971593,'HOCl_2':53.968644,
               'N2_1':28.006148,'N2_2':29.003182,
               'HCN_1':27.010899,'HCN_2':28.014254,'HCN_3':28.007933,
               'CH3Cl_1':49.992328,'CH3CL_2':51.989379,
               'H2O2_1':34.00548,
               'C2H2_1':26.01565,'C2H2_2':27.019005,'C2H2_3':27.021825,
               'C2H6_1':30.04695,'C2H6_2':31.050305,
               'PH3_1':33.997238,
               'COF2_1':65.991722,'COF2_2':66.995083,
               'SF6_1':145.962492,
               'H2S_1':33.987721,'H2S_2':35.983515,'H2S_3':34.987105,
               'HCOOH_1':46.00548,
               'HO2_1':32.997655,
               'O_1':15.994915,
               'ClONO2_1':96.956672,'ClONO2_2':98.953723,
               'NO+_1':29.997989,
               'HOBr_1':95.921076,'HOBr_2':97.919027,
               'C2H4_1':28.0313,'C2H4_2':29.034655,
               'CH3OH_1':32.026215,
               'CH3Br_1':93.941811,'CH3Br_2':95.939764,
               'CH3CN_1':41.026549,
               'CF4_1':87.993616,
               'C4H2_1':50.01565,
               'HC3N_1':51.010899,
               'H2_1':2.01565,'H2_2':3.021825,
               'CS_1':43.971036,'CS_2':45.966787,'CS_3':44.974368,'CS_4':44.970399,
               'SO3_1':79.95682,
               'C2N2_1':52.006148,
               'COCl2_1':97.9326199796,'COCl2_2':99.9296698896,
               'CS2_1':75.94414,'CS2_2':77.93994,'CS2_3':76.943256,'CS2_4':76.947495}
 
    return mass[mol_isot_code]

