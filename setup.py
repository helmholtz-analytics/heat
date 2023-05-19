from setuptools import setup, find_packages
import codecs


with codecs.open("README.md", "r", "utf-8") as handle:
    long_description = handle.read()

__version__ = None  # appeases flake, assignment in exec() below
with open("./heat/core/version.py") as handle:
    exec(handle.read())

setup(
    name="heat",
    packages=find_packages(exclude=("*tests*", "*benchmarks*")),
    package_data={"heat.datasets": ["*.csv", "*.h5", "*.nc"]},
    version=__version__,
    description="A framework for high-performance data analytics and machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Helmholtz Association",
    author_email="martin.siggel@dlr.de",
    url="https://github.com/helmholtz-analytics/heat",
    keywords=["data", "analytics", "tensors", "distributed", "gpu"],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=[
        "mpi4py>=3.0.0",
        "numpy>=1.20.0",
        "torch>=1.8.0, <2.0.2",
        "scipy>=0.14.0",
        "pillow>=6.0.0",
        "torchvision>=0.8.0",
    ],
    extras_require={
        "docutils": ["docutils>=0.16"],
        "hdf5": ["h5py>=2.8.0"],
        "netcdf": ["netCDF4>=1.5.6"],
        "dev": ["pre-commit>=1.18.3"],
        "examples": ["scikit-learn>=0.24.0", "matplotlib>=3.1.0"],
        "cb": ["perun>=0.2.0"],
    },
)
