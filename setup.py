from setuptools import setup, find_packages


with open("README.md", "r") as handle:
    long_description = handle.read()

__version__ = None  # appeases flake, assignment in exec() below
with open("./heat/core/version.py") as handle:
    exec(handle.read())

setup(
    name="heat",
    packages=find_packages(exclude=("*tests*",)),
    data_files=["README.md", "LICENSE"],
    version=__version__,
    description="A framework for high performance data analytics and machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Helmholtz Association",
    author_email="martin.siggel@dlr.de",
    url="https://github.com/helmholtz-analytics/heat",
    keywords=["data", "analytics", "tensors", "distributed", "gpu"],
    python_requires="~=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=["mpi4py>=3.0.0", "numpy>=1.13.0", "torch>=1.5.0"],
    extras_require={
        "hdf5": ["h5py>=2.8.0"],
        "netcdf": ["netCDF4>=1.4.0,<=1.5.2"],
        "dev": ["pre-commit>=1.18.3"],
    },
)
