import numpy as np
import emcee
import pandas as pd
from IPython.display import display, Math
import corner
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.constants import c,h, k_B, G, M_sun, au, pc, u
from astropy.table import Table
from astropy import units as un
from astropy.convolution import Gaussian1DKernel, convolve

from spectools_ir.utils import extract_hitran_data, get_global_identifier, translate_molecule_identifier, get_molmass

def compute_model_fluxes(mydata,samples):
    '''
    Function to compute model fluxes for same lines as in dataset.

    Parameters
    ----------
    mydata : Data object
     Instance of Data class, provides routine with lines to calculate
    samples : numpy array
     Array holding arrays of MCMC output samples.  Used to find best-fit parameters.s

    Returns
    ---------
    lineflux : numpy array
     Array of line fluxes 
    '''

    bestfit_dict=find_best_fit(samples)
    logn=bestfit_dict['logN']
    temp=bestfit_dict['T']
    logomega=bestfit_dict['logOmega']

    omega=10**logomega
    n_col=10**logn
    si2jy=1e26   #SI to Jy flux conversion factor 
#If local velocity field is not given, assume sigma given by thermal velocity
    mu=u.value*mydata.molmass
    deltav=np.sqrt(k_B.value*temp/mu)   #m/s
#Use line_ids to extract relevant HITRAN data
    wn0=mydata.wn0
    aup=mydata.aup
    eup=mydata.eup
    gup=mydata.gup
    eup_k=mydata.eup_k
    elower=mydata.elower

#Compute partition function                    
    q=_get_partition_function(mydata,temp)
#Begin calculations
    afactor=((aup*gup*n_col)/(q*8.*np.pi*(wn0)**3.)) #mks
    efactor=h.value*c.value*eup/(k_B.value*temp)
    wnfactor=h.value*c.value*wn0/(k_B.value*temp)
    phia=1./(deltav*np.sqrt(2.0*np.pi))
    efactor2=eup_k/temp
    efactor1=elower*1.e2*h.value*c.value/k_B.value/temp
    tau0=afactor*(np.exp(-1.*efactor1)-np.exp(-1.*efactor2))*phia  #Avoids numerical issues at low T 
    w0=1.e6/wn0

    dvel=0.1e0    #km/s                                     
    nvel=1001
    vel=(dvel*(np.arange(0,nvel)-500.0))*1.e3     #now in m/s
#Now loop over transitions and velocities to calculate flux 
    tau=np.exp(-vel**2./(2.*np.vstack(deltav)**2.))*np.vstack(tau0)

#Create array to hold line fluxes (one flux value per line) 
    nlines=np.size(tau0)
    f_arr=np.zeros([nlines,nvel])     #nlines x nvel
    lineflux=np.zeros(nlines)

    for i in range(nlines):  #I might still be able to get rid of this loop 
        f_arr[i,:]=2*h.value*c.value*wn0[i]**3./(np.exp(wnfactor[i])-1.0e0)*(1-np.exp(-tau[i,:]))*si2jy*omega
        lineflux_jykms=np.sum(f_arr[i,:])*dvel
        lineflux[i]=lineflux_jykms*1e-26*1.*1e5*(1./(w0[i]*1e-4))    #mks  
    return lineflux


def get_samples(chain,burnin):
    '''                                                                                                                          
    Function to remove burnin samples from MCMC output chain, reconfigure to make samples
                                                                                                                                 
    Parameters                                                                                                                   
    ----------                                                                                                                   
    presamples : numpy array                                                                                                     
     Array holding arrays of MCMC output samples.  Used to find best-fit parameters.s                                            
    burnin : integer                                                                                                             
     Number of samples considered part of the "burnin"                                                                           
                                                                                                                                 
    Returns                                                                                                                      
    ---------                                                                                                                    
    postsamples : numpy array                                                                                                    
     Array holding arrays of MCMC output samples after removal of burnin samples.                                                
    '''
    ndims = chain.shape[2]
    samples = chain[:, burnin:, :].reshape((-1, ndims))
    
    return samples

def _get_partition_function(mydata,temp):
    q=np.zeros(mydata.nlines)
    for myunique_id in mydata.unique_globals:
        myq=mydata.qdata_dict[str(myunique_id)][int(temp)-1]
        mybool=(mydata.global_id == myunique_id)
        q[mybool]=myq
    return q

def corner_plot(samples,outfile=None,**kwargs):
    '''
    Function to make a corner plot for output samples

    Parameters
    ----------
    samples : numpy array
     Array holding arrays of MCMC output samples.
    outfile : str, optional
     Path to output file to hold resultant figure
    '''


    parlabels=[ r"$\log(\ n_\mathrm{tot} [\mathrm{m}^{-2}]\ )$",r"Temperature [K]", "$\log(\ {\Omega [\mathrm{rad}]}\ )$"]
    fig = corner.corner(samples,
                    labels=parlabels,
                    show_titles=True, title_kwargs={"fontsize": 12},**kwargs)
    if(outfile is not None):
        fig.savefig(outfile)

def trace_plot(samples,xr=[None,None]):
    '''
    Function to make a trace plot for each parameter of slab model fit

    Parameters
    ----------
    samples : numpy array
     Array holding arrays of MCMC output samples. 
    xr : 2-element array, optional
     Range of samples to plot
    '''

    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    parlabels=[ r"$\log(\ n_\mathrm{tot} [\mathrm{m}^{-2}]\ )$",r"Temperature [K]", "$\log(\ {\Omega [\mathrm{rad}]}\ )$"]
    ndims=3
    for i in range(ndims):
        ax = axes[i]
        ax.plot(samples[:,i], "k", alpha=0.3)    #0th walker, i'th dimension
        ax.set_ylabel(parlabels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_xlim(xr)
    axes[-1].set_xlabel("step number");

def find_best_fit(samples,show=False):
    '''
    Function to find best fit parameters

    Parameters
    ----------
    samples : numpy array
     Array holding arrays of MCMC output samples.
    show : boolean, optional
     Boolean to display or not display nicely-formatted results

    Returns
    ---------
    bestfit_dict : 
     Dictionary holding best-fit slab model parameters with plus and minus error bars.
     Based on 16, 50 and 84th percentiles of posterior distribution.
    '''
    parlabels=[ r"\log(\ n_\mathrm{tot} [\mathrm{m}^{-2}]\ )",r"Temperature [K]", r"\log(\ {\Omega [\mathrm{rad}]}\ )"]
    paramkeys=['logN','T','logOmega']
    perrkeys=['logN_perr','T_perr','logOmega_perr']
    nerrkeys=['logN_nerr','T_nerr','logOmega_nerr']
    bestfit_dict={}
    for i in range(3):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], parlabels[i])
        if(show==True): display(Math(txt))
        bestfit_dict[paramkeys[i]]=mcmc[1]
        bestfit_dict[perrkeys[i]]=q[1]     
        bestfit_dict[nerrkeys[i]]=q[0]    

    return bestfit_dict
