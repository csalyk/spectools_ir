import numpy as np
import urllib
import emcee
import pandas as pd
import json as json
import time
import pkgutil
from scipy.interpolate import interp1d
import pdb as pdb
from astropy.io import fits
from astropy.constants import c,h, k_B, G, M_sun, au, pc, u
from astropy.table import Table
from astropy import units as un
from astropy.convolution import Gaussian1DKernel, convolve

from spectools_ir.utils import extract_hitran_data, get_global_identifier, translate_molecule_identifier, get_molecule_identifier, get_molmass, spec_convol

def read_data_from_fits(fitsfile,wavename='WAVELENGTH',fluxname='FLUX',errname='UNCERTAINTY',waveunits='micron',fluxunits='Jy'):
    hdulist=fits.open(fitsfile)
    fitsdata=hdulist[1].data
    data=Table()
    
    if(waveunits=='nm'): data['wave']=fitsdata[wavename][0]*1e-3
    if(waveunits=='micron'): data['wave']=fitsdata[wavename][0]

    if(fluxunits=='Jy'): 
        data['flux']=fitsdata[fluxname][0]
        data['error']=fitsdata[errname][0]
    if(fluxunits=='mJy'): 
        data['flux']=fitsdata[fluxname][0]*1e-3
        data['error']=fitsdata[errname][0]*1e-3
        
    return data

class Config():
    '''
    Class for handling input parameters
    '''
    def __init__(self,config_file=None):
        if(config_file is None):
            config_data=pkgutil.get_data(__name__,'config.json')
            self.config=json.loads(config_data)
        if(config_file is not None):
            with open(config_file, 'r') as file:
                self.config = json.load(file)
        self.config['molmass']=get_molmass(self.config['molecule'],isotopologue_number=self.config['iso'])
                
    def getpar(self,name):
        return self.config[name]

    def display(self):
 
        print(json.dumps(self.config, indent=1))

class Retrieval():
    '''
    Class for handling the Bayesian retrieval using the emcee package.
    
    '''
    def __init__(self,Config,SpecData):
        self.Config = Config
        self.SpecData = SpecData
        
        self.wmax=np.nanmax(self.SpecData.wave)
        self.wmin=np.nanmin(self.SpecData.wave)
        try:
            hitran_data = extract_hitran_data(self.Config.getpar('molecule'),self.wmin,self.wmax,isotopologue_number=self.Config.getpar('iso'))
        except:
            print("astroquery call to HITRAN failed. This can happen when your molecule does not have any lines in the requested wavelength region")

        self.wn0=hitran_data['wn']*1e2 # now m-1  
        self.aup=hitran_data['a']
        self.eup=(hitran_data['elower']+hitran_data['wn'])*1e2 #now m-1 
        self.gup=hitran_data['gp']
        self.eup_k=(hitran_data['elower']+hitran_data['wn'])*1e2*h.value*c.value/k_B.value
        self.elower=hitran_data['elower'] 
        self.molec_id=hitran_data['molec_id']
        self.nlines = len(hitran_data['elower'])
        self.local_iso_id=hitran_data['local_iso_id']
        self.global_id=self._return_global_ids()
        self.unique_globals = np.unique(self.global_id)
        self.qdata_dict=self._get_qdata()
        
    #Returns HITRAN global IDs for all lines
    def _return_global_ids(self):
        global_id = np.array([get_global_identifier(translate_molecule_identifier(self.molec_id[i]), isotopologue_number=self.local_iso_id[i]) for i in np.arange(self.nlines)])
        return global_id        

    def _get_qdata(self):
        id_array=self.unique_globals
        q_dict={}
        for myid in id_array:
            qurl='https://hitran.org/data/Q/'+'q'+str(myid)+'.txt'
            handle = urllib.request.urlopen(qurl)
            qdata = pd.read_csv(handle,sep=' ',skipinitialspace=True,names=['temp','q'],header=None)
            print('Reading partition function from: ',qurl)
            q_dict.update({str(myid):qdata['q']})
        return q_dict
    
    def run_emcee(self):
        #Initialize walkers
        lognini = np.random.uniform(self.Config.getpar('lognmin'), self.Config.getpar('lognmax'), self.Config.getpar('Nwalkers')) # initial logn points 
        tini = np.random.uniform(self.Config.getpar('tmin'), self.Config.getpar('tmax'), self.Config.getpar('Nwalkers')) # initial logn points 
        logomegaini = np.random.uniform(self.Config.getpar('logomegamin'), self.Config.getpar('logomegamax'), self.Config.getpar('Nwalkers')) # initial logn points 
        inisamples = np.array([lognini, tini, logomegaini]).T
        ndims = inisamples.shape[1] 
        Nwalkers=self.Config.getpar('Nwalkers')
        Nsamples=self.Config.getpar('Nsamples')
        Nburnin=self.Config.getpar('Nburnin')
        sampler = emcee.EnsembleSampler(Nwalkers, ndims, self._lnposterior)

        start_time=time.time()
        sampler.run_mcmc(inisamples, Nsamples+Nburnin)
        end_time=time.time()
        print("Number of total samples:", Nwalkers*Nsamples)
        print("Run time [s]:", end_time-start_time)
        #sampler.chain has dimensions (Nwalkers, Nburnin+Nsamples,ndims)
#        samples = sampler.chain[:, Nburnin:, :].reshape((-1, ndims))  

        return sampler

    def _lnprior(self, theta):
        lp = 0.  #initialize log prior
        logn, temp, logomega = theta # unpack the model parameters from the list                                             
        #First parameter: logn - uniform prior
        lognmin = self.Config.getpar('lognmin')  # lower range of prior                                                        
        lognmax = self.Config.getpar('lognmax')  # upper range of prior                                                        
        lp = 0. if lognmin < logn < lognmax else -np.inf 
        #Second parameter: temperature - uniform prior
        tmin = self.Config.getpar('tmin')
        tmax = self.Config.getpar('tmax')
        lpt = 0. if tmin < temp < tmax else -np.inf
        lp += lpt #Add log prior due to temperature to lp due to logn
        #Third parameter: Omega - uniform prior
        logomegamin = self.Config.getpar('logomegamin')
        logomegamax = self.Config.getpar('logomegamax')
        lpo = 0. if logomegamin < logomega < logomegamax else -np.inf
        lp += lpo #Add log prior due to omega to lp due to temperature,logn

        return lp

    def _lnlikelihood(self, theta):
        md = self._compute_spectrum(theta)
        data=self.SpecData.flux
        sigma=self.SpecData.error
        isfin=(np.isfinite(data) & np.isfinite(sigma))
        
        lnlike = -0.5*np.sum(((md[isfin] - data[isfin])/sigma[isfin])**2)

        return lnlike

    def _lnposterior(self,theta):
        lp = self._lnprior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self._lnlikelihood(theta)

    def _compute_spectrum(self,theta):
        logn, temp, logomega = theta    #unpack parameters
        omega=10**logomega
        n_col=10**logn
        si2jy=1e26   #SI to Jy flux conversion factor

#If local velocity field is not given, assume sigma given by thermal velocity

        mu=u.value*self.Config.getpar('molmass')
        deltav=np.sqrt(k_B.value*temp/mu)   #m/s 

        wn0=self.wn0
        aup=self.aup
        eup=self.eup
        gup=self.gup
        eup_k=self.eup_k
        elower=self.elower
        
#Compute partition function
        q=self._get_partition_function(temp) #Fix later
#Begin calculations
       
        afactor=((aup*gup*n_col)/(q*8.*np.pi*(wn0)**3.)) #mks                                                                 
        efactor=h.value*c.value*eup/(k_B.value*temp)
        wnfactor=h.value*c.value*wn0/(k_B.value*temp)
        phia=1./(deltav*np.sqrt(2.0*np.pi))
        efactor2=eup_k/temp
        efactor1=elower*1.e2*h.value*c.value/k_B.value/temp
        tau0=afactor*(np.exp(-1.*efactor1)-np.exp(-1.*efactor2))*phia  #Avoids numerical issues at low T

        oversamp = 3
        dvel = deltav/oversamp    #m/s                                                                                      
        nvel = 10*oversamp+1 #5 sigma window                                                                                
        vel = (dvel*(np.arange(0,nvel)-(nvel-1)/2))

    #Now loop over transitions and velocities to calculate flux  
        nlines=np.size(tau0)
        tau = np.zeros([nlines,nvel])
        wave = np.zeros([nlines,nvel])
        for ha,mytau in enumerate(tau0):
            tau[ha,:] = tau0[ha]*np.exp(-vel**2./(2.*deltav**2.))
            wave[ha,:] = 1.e6/wn0[ha]*(1+vel/c.value)

#Now interpolate over wavelength space so that all lines can be added together                                      
        w_arr = wave            #nlines x nvel                                                                              
        f_arr = w_arr-w_arr     #nlines x nvel                                                                              
        nbins = int(oversamp*(self.wmax-self.wmin)/self.wmax*(c.value/deltav))

#Create arrays to hold full spectrum (optical depth vs. wavelength)                                                 
        totalwave = np.logspace(np.log10(self.wmin-10*deltav/c.value*self.wmax),np.log10(self.wmax+10*deltav/c.value*self.wmax),nbins) #Extend beyond input wave by 10xdelta_wave
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
        convol_fwhm = self.Config.getpar('fwhm')
        if(convol_fwhm is not None):
            convolflux = spec_convol(wave,flux,convol_fwhm)

        convolflux_interp=np.interp(self.SpecData.wave,wave,convolflux)

        return convolflux_interp
#------------------------------------------------------------------------------
    def _get_partition_function(self,temp):
#Loop through each unique identifier
#For each unique identifier, assign q values accordingly
        q=np.zeros(self.nlines)
        for myunique_id in self.unique_globals:
            myq=self.qdata_dict[str(myunique_id)][int(temp)-1]  #Look up appropriate q value
            mybool=(self.global_id == myunique_id)              #Find where global identifier equals this one
            q[mybool]=myq                                      #Assign q values where global identifier equals this one
        return q
#------------------------------------------------------------------------------------                                     
class SpecData():
    def __init__(self,data):
        self.wave=data['wave'] #microns
        self.flux=data['flux'] #Jy
        self.error=data['error'] #Jy

#---------------------
    #Returns HITRAN molecular masses for all lines
    def _return_molmasses(self):
        molmass_arr = np.array([get_molmass(translate_molecule_identifier(self.molec_id[i]), isotopologue_number=self.local_iso_id[i]) for i in np.arange(self.nlines)])
        return molmass_arr
#---------------------


#------------------------------------------------------------------------------------                                     
    def rot_diagram(self,units='mks',modelfluxes=None):
        x=self.eup_k
        mywn0=self.wn0
        y=np.log(self.lineflux/(mywn0*self.gup*self.aup))  #All mks, so wn in m^-1
        if(units=='cgs'):
            y=np.log(1000.*self.lineflux/((self.wn0*1e-2)*self.gup*self.aup))   #All cgs
        if(units=='mixed'):
            y=np.log(self.lineflux/((self.wn0*1e-2)*self.gup*self.aup))
        rot_dict={'x':x,'y':y,'units':units}
        if(modelfluxes is not None):
            rot_dict['modely']=np.log(modelfluxes/(mywn0*self.gup*self.aup))  #All mks, so wn in m^-1
            if(units=='cgs'):
                rot_dict['modely']=np.log(modelfluxes*1000./(self.wn0*1e-2*self.gup*self.aup))  #All cgs
            if(units=='mixed'):
                rot_dict['modely']=np.log(modelfluxes/(self.wn0*1e-2*self.gup*self.aup))  #Mixed units

        return rot_dict
#------------------------------------------------------------------------------------                                     
