from distutils.core import setup
setup(
    name = 'spectools-ir',
    packages = ['flux_calculator','slabspec','slab_fitter'],
    version = '0.1.0',
    description = 'Tools for analysis and modeling of IR spectra',
    long_description=open('README.md').read(),
    author='Colette Salyk',
    author_email='cosalyk@vassar.edu',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
        ],
    include_package_data=True,    
    package_data={'': ['*.json']},
    install_requires=[
        'astropy',
        'astroquery',
        'emcee',
        'numpy',
        'os',
        'urllib',
        'emcee',
        'pandas',
        'json',
        'time',
        'pkgutil',
        'IPython',
        'corner',
        'matplotlib'
    ]
) 
