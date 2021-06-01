# spectools-ir
Spectools_ir is a small suite of tools designed for analysis of medium/high-resolution IR molecular astronomical spectra.  It consists
of three main sub-modules (flux_calculator, slabspec, and slab_fitter) as well as a 'utils' sub-module, with
a few additional functions.  

Spectools_ir was written with infrared medium/high-resolution molecular spectroscopy in mind.  It often assumes spectra
are in units of Jy and microns, and it uses information from the HITRAN molecular database.  Some routines are more general, but
users interested in other applications should proceed with caution.

Users are requested to let the developer know if they are using the code in spectools_ir.  The code has been
tested for only a few use cases, and users utilize at their own risk.

# Requirements
Requires internet access to utilize astroquery.hitran and access HITRAN partition function files.

Requires several standard scientific packages, as well as astropy, astroquery, corner, and emcee.

# Modules
flux_calculator is a set of python codes to compute line fluxes from an IR spectrum.

slabspec is a set of python codes to produce LTE slab model emission spectra of molecules using the HITRAN database.

slab_fitter is a set of python codes to perform MCMC slab model fits to line fluxes using "emcee" (Foreman-Mackey et al. 2013; https://github.com/dfm/emcee; https://emcee.readthedocs.io/en/stable/) with flat priors. 

utils contains some useful utility functions, including functions specific to JWST MIRI-MRS.

# Usage

Example usage can be found at https://github.com/csalyk/spectools_ir/ in docs/example.ipynb

# License
[MIT](https://choosealicense.com/licenses/mit/)

