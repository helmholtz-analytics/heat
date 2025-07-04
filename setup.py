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
    package_data={"heat.datasets": ["*.csv", "*.h5", "*.nc"], "heat": ["py.typed"]},
    version=__version__,
    description="A framework for high-performance data analytics and machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Helmholtz Association",
    author_email="martin.siggel@dlr.de",
    url="https://github.com/helmholtz-analytics/heat",
    keywords=["data", "analytics", "tensors", "distributed", "gpu"],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=[
        "mpi4py>=3.0.0",
        "numpy>=1.23.5",
        "torch>=2.0.0, <2.7.2",
        "scipy>=1.14.0",
        "pillow>=6.0.0",
        "torchvision>=0.15.2, <0.22.2",
    ],
    extras_require={
        # Dev
        "dev": ["pre-commit>=1.18.3"],
        # CI/CB
        "cb": ["perun>=0.8"],
        # Examples/ Tutorial
        "examples": ["scikit-learn>=0.24.0", "matplotlib>=3.1.0", "ipyparallel", "jupyter"],
        # IO
        "pandas": ["pandas>=1.4"],
        "hdf5": ["h5py>=2.8.0"],
        "netcdf": ["netCDF4>=1.5.6"],
        "zarr": ["zarr<3.0.9"],
        # Docs
        "docs": [
            "sphinx",
            "sphinx_rtd_theme",
            "sphinx-autoapi",
            "nbsphinx",
            "sphinx-autobuild",
            "sphinx-copybutton",
        ],
    },
)
