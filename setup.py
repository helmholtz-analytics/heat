from distutils.core import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='heat',
    packages=['heat', 'heat.core', 'heat.ml', 'heat.ml.cluster'],
    data_files=['README.md', 'LICENSE'],
    version='0.0.3',
    description='A framework for high performance data analytics and machine learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Helmholtz Association',
    author_email='martin.siggel@dlr.de',
    url='https://github.com/helmholtz-analytics/heat',
    keywords=['data', 'analytics', 'tensors', 'distributed', 'gpu'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3.5',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
    ],
    install_requires=[
        'mpi4py>=3.0.0',
        'numpy>=1.13.0',
        'torch>=1.0.0'
    ],
    extra_require={
        'hdf5':  ['h5py>=2.8.0'],
        'netcdf': ['netCDF4>=1.4.0'],
    }
)
