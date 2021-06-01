import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from astropy.constants import c
from astropy.table import Table

from spectools_ir.utils import extract_hitran_data, fwhm_to_sigma, sigma_to_fwhm
from .helpers import _line_fit, _calc_linewidth, _calc_line_flux_from_fit
from .helpers import _strip_superfluous_hitran_data, _convert_quantum_strings
from .helpers import _calc_numerical_flux

def calc_fluxes(wave,flux,hitran_data, fwhm_v=20., sep_v=40.,cont=1.,verbose=True,vet_fits=False,
                plot=False,v_dop=0,amp=0.1,ymin=None,ymax=None):
    '''                                                                                     
                                                                                            
    Parameters                                                                              
    ---------                                                                               
    wave : numpy array                                                                      
        wavelength values, in microns                                                       
    flux : numpy array                                                                      
        flux density values, in units of Jy                                
    htiran_data : astropy table
        output from extract_hitran
    fwhm_v : float, optional - defaults to 8 km/s
        estimate of line width in km/s for line fitting input
    sep_v : float, optional - defaults to 40 km/s
        total width used for line fits in km/s
    amp : float, optional - defaults to 0.1
        estimated amplitude of Gaussian fit
    ymin : float, optional - defaults to 0+continuum
        minimum of y axis for plotting
    ymax : float, optional - defaults to 2+continuum
        maximum of y axis for plotting
    cont : float, optional - defaults to 1.
        Continuum level, in Jy.
    verbose: bool, optional - defaults to True
        True prints out some messages during runtime.
    vet_fits: bool, optional - defaults to False
        If True, user is prompted to decide if fit is good or not.
    plot: bool, optional - defaults to False
        If True, data and fits are plotted.  If vet_fits=True, this gets set to True automatically.
    v_dop : float, optional (defaults to 0)
        Doppler shift in km/s of spectrum relative to vacuum.  Note that this makes no assumptions about
         reference frame.

    Returns                                                                                 
    --------                                                                                
    lineflux_data : astropy table
       Table holding both HITRAN data and fit parameters (including flux, line width, and Doppler shift) 
        for fit lines.  
                                                                                            
    '''
    if(vet_fits==True): 
        plot=True
    lineflux_data=_convert_quantum_strings(hitran_data)
    lineflux_data=_strip_superfluous_hitran_data(lineflux_data)

    nlines=np.size(lineflux_data)
    #Add new columns to astropy table to hold line fluxes and error bars
    lineflux_data['lineflux']=np.zeros(nlines)  
    lineflux_data['lineflux_Gaussian']=np.zeros(nlines)  
    lineflux_data['lineflux_err']=np.zeros(nlines)  
    lineflux_data['linewidth']=np.zeros(nlines)
    lineflux_data['linewidth_err']=np.zeros(nlines)
    lineflux_data['v_dop_fit']=np.zeros(nlines)  
    lineflux_data['v_dop_fit_err']=np.zeros(nlines)  
    lineflux_data['continuum']=np.zeros(nlines)  
    lineflux_data['continuum_err']=np.zeros(nlines)  
    goodfit_bool=[True]*nlines
    #Loop through HITRAN wavelengths
    for i,w0 in enumerate(lineflux_data['wave']):   
        #Perform Gaussian fit for each line
        #Calculate Doppler shift, line width, and line separation in microns
        wdop=v_dop*1e3/c.value*w0
        dw=sep_v*1e3/c.value*w0
        dw2=2*sep_v*1e3/c.value*w0
        sig_w=fwhm_to_sigma(fwhm_v*1e3/c.value*w0)
        mybool=((wave>(w0+wdop-dw)) & (wave<(w0+wdop+dw)) & np.isfinite(flux))
        myx=wave[mybool]
        myy=flux[mybool]
        if((len(myx) <= 5) & (verbose==True) ):
            print('Not enough data near ', w0+wdop, ' microns. Skipping.')
            goodfit_bool[i]=False
        if(len(myx) > 5):
            g=_line_fit(myx,myy,nterms=4,p0=[amp,w0+wdop,sig_w,cont])
            if(g!=-1):   #curve fit succeeded
                p=g['parameters']
                perr=g['parameter_errors']
                resid=g['resid']
                sigflux=np.sqrt(np.mean(resid**2.))
                (lineflux,lineflux_err)=_calc_line_flux_from_fit(p,sigflux=sigflux)
                lineflux_data['lineflux_Gaussian'][i]=lineflux.value
                lineflux_data['lineflux_err'][i]=lineflux_err.value
                lineflux_num=_calc_numerical_flux(myx,myy,p)
                lineflux_data['lineflux'][i]=lineflux_num.value
                lineflux_data['linewidth'][i]=np.abs((_calc_linewidth(p,perr=perr))[0].value)
                lineflux_data['linewidth_err'][i]=np.abs((_calc_linewidth(p,perr=perr))[1].value)
                lineflux_data['v_dop_fit'][i]=(p[1]-w0)/w0*c.value*1e-3   #km/s
                lineflux_data['v_dop_fit_err'][i]=(perr[1])/w0*c.value*1e-3   #km/s
                lineflux_data['continuum'][i]=(p[3])   #Jy
                lineflux_data['continuum_err'][i]=(perr[3])   #Jy

                if(plot==True):
                    fig=plt.figure(figsize=(10,3))
                    ax1=fig.add_subplot(111)
                    ax1.plot(wave,flux,'C0',drawstyle='steps-mid',label='All data')
                    ax1.plot(myx,myy,'C1',drawstyle='steps-mid',label='Fit data')
                    ax1.plot(myx,g['yfit'],'C2',label='Fit')
                    ax1.axvline(w0+wdop,color='C3',label='Line center')
                    ax1.set_xlim(np.min(myx)-dw2,np.max(myx)+dw2)
                    if(ymin is None): ymin=np.min(myy)
                    if(ymax is None): ymax=2+np.min(myy)
                    ax1.set_ylim(ymin,ymax)
                    ax1.set_xlabel(r'Wavelength [$\mu$m]')
                    ax1.set_ylabel(r'F$_\nu$ [Jy]')
                    ax1.axvline(p[1]-3*p[2],label='Numerical limit',color='C3',linestyle='--')
                    ax1.axvline(p[1]+3*p[2],color='C3',linestyle='--')
                    ax1.legend()
                    if('Qpp' in hitran_data.columns): ax1.set_title(hitran_data['Qpp'][i])
                    plt.show(block=False)
                    plt.close()    
                user_input=None
                if(vet_fits==True):
                    user_input=input("Is this fit okay? [y or n]")
                    while((user_input!='y') & (user_input!='n')):
                        user_input=input("Is this fit okay? Please enter y or n.") 
                if(user_input=='n'): 
                    goodfit_bool[i]=False
            if(g==-1):   #curve fit failed
                goodfit_bool[i]=False
                if(plot==True):
                    fig=plt.figure(figsize=(10,3))
                    ax1=fig.add_subplot(111)
                    ax1.plot(wave,flux,'C0',drawstyle='steps-mid',label='All data')
                    ax1.plot(myx,myy,'C1',drawstyle='steps-mid',label='Fit data')
                    ax1.axvline(w0+wdop,color='C3',label='Line center')
                    ax1.set_xlim(np.min(myx)-dw2,np.max(myx)+dw2)
                    if(ymin is None): ymin=np.min(myy)
                    if(ymax is None): ymax=2+np.min(myy)
                    ax1.set_ylim(ymin,ymax)
                    ax1.set_xlabel(r'Wavelength [$\mu$m]')
                    ax1.set_ylabel(r'F$_\nu$ [Jy]')
                    ax1.legend()
                    if('Qpp' in hitran_data.columns): ax1.set_title(hitran_data['Qpp'][i])
                    plt.show(block=False)
                    plt.pause(0.5)
                    plt.close()    

    lineflux_data['lineflux'].unit = 'W / m2'
    lineflux_data['lineflux_err'].unit = 'W / m2'
    lineflux_data['linewidth'].unit = 'km / s'
    lineflux_data['linewidth_err'].unit = 'km / s'
    lineflux_data['v_dop_fit'].unit = 'km / s'
    lineflux_data['v_dop_fit_err'].unit = 'km / s'
    lineflux_data['continuum'].unit = 'Jy'
    lineflux_data['continuum_err'].unit = 'Jy'

    lineflux_data=lineflux_data[goodfit_bool]

    return lineflux_data
    

def make_lineshape(wave,flux, lineflux_data, dv=3., voffset=None,norm=None):
    '''                                                                                                                                                   
                                                                                                                                                          
    Parameters                                                                                                                                            
    ---------                                                                                                                                             
    wave : numpy array                                                                                                                                    
        set of wavelengths for spectrum, in units of microns                                                                                              
    flux : numpy array                                                                                                                                    
        set of fluxes for spectrum, in units of Jy                                                                                                        
    lineflux_data : astropy table                                                                                                                         
        table in same format as flux_calculator output                                                                                                    
    dv : float, optional
        bin size for resultant lineshape, in km/s.  Defaults to 3 km/s.
    voffset : float, optional                                                                                                                             
        Doppler shift of observed spectrum in km/s.  Defaults to median of lineflux fits.                                                                 
    norm : str, optional
        String describing normalization type.  Currently only option is 'Maxmin', which sets max to 1, min to 0.  Defaults to None.
    
    Returns
    ---------                                                                                                                                             
    (interpvel,interpflux): tuple containing interpolated line shape     
                                                                                                                                                          
    '''
    w0=np.array(lineflux_data['wave'])

    nlines=np.size(w0)

    if(voffset is None and 'v_dop_fit' in lineflux_data.columns): 
        voffset=np.median(lineflux_data['v_dop_fit'])    #If Doppler shift is not specified, use median from lineflux_data if it exists
    if(voffset is None and not('v_dop_fit' in lineflux_data.columns)): 
        voffset=0    #If Doppler shift is not defined, use 0 if lineflux_data has no element v_dop_fit  
    w0*=(1+voffset*1e3/c.value)    #Apply Doppler shift                                                                                                   

    #Make interpolation grid                                                                                                                              
    nvels=151
    nlines=np.size(w0)
    interpvel=np.arange(nvels)*dv-75.*dv
    interpind=np.zeros((nvels,nlines))+1  #keeps track of weighting for each velocity bin                                                                 
    interpflux=np.zeros((nvels,nlines))

    #Loop through all w0 values                                                                                                                           
    for i,my_w0 in enumerate(w0):
        mywave = wave[(wave > (my_w0-0.003)) & (wave < (my_w0+0.003))]  #Find nearby wavelengths                                                          
        myflux = flux[(wave > (my_w0-0.003)) & (wave < (my_w0+0.003))]  #Find nearby fluxes                                                               
        myvel = c.value*1e-3*(mywave - my_w0)/my_w0                     #Convert wavelength to velocity                                                   
        f1=interp1d(myvel, myflux, kind='linear', bounds_error=False)   #Interpolate onto velocity grid                                                   
        interpflux[:,i]=f1(interpvel)
        w=np.where((interpvel > np.max(myvel)) | (interpvel < np.min(myvel)) | (np.isfinite(interpflux[:,i]) != 1 )  ) #remove fluxes beyond edges, NaNs  
        if(np.size(w) > 0):
            interpind[w,i]=0
            interpflux[w,i]=0
    numer=np.nansum(interpflux,1)
    denom=np.nansum(interpind,1)
    mybool=(denom==0)   #Find regions where there is no data                                                                                              
    numer[mybool]='NaN' #Set to NaN                                                                                                                       
    denom[mybool]=1
    interpflux=numer/denom

    if(norm=='Maxmin'):  #Re-normalize if desired                                                                                                         
        interpflux=(interpflux-np.nanmin(interpflux))/np.nanmax(interpflux-np.nanmin(interpflux))

    return (interpvel,interpflux)

