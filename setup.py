from setuptools import setup
setup(
    name = 'spectools-ir',
    packages = ['spectools_ir','spectools_ir.slabspec','spectools_ir.flux_calculator','spectools_ir.utils','spectools_ir.slab_fitter'],
    version = '0.1.7',
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
        'astropy>=4.2',
        'emcee>=3.0',
        'numpy>=1',
        'pandas>=0.24',
        'corner>=2.1',
        'matplotlib>=3',
        'astroquery>0.4'
    ]
) 
