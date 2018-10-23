from distutils.core import setup

setup(
    name='heat',
    packages=['heat'],
    version='0.0.1',
    description='A framework for high performance data analytics and machine learning.',
    author='Helmholtz Association',
    author_email='martin.siggel@dlr.de',
    url='https://github.com/helmholtz-analytics/heat',
    #  download_url = 'https://github.com/helmholtz-analytics/heat/archive/0.1.tar.gz', # TBD
    keywords=['data', 'analytics', 'tensors', 'distributed', 'gpu'],
    classifiers=[],
    install_requires=[
        'mpi4py>=3.0.0',
        'numpy>=1.13.0',
        'torch>=0.4.0',
    ],
    extras_require={
        'hdf5': ['h5py>=2.8.0'],
        'netcdf': ['netCDF4>=1.4.0']
    }
)
