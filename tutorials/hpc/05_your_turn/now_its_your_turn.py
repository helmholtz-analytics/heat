import heat as ht
import numpy as np
import h5py

# Now its your turn! Download one of the following three data sets and play around with it.
# Possible ideas:
# get familiar with the data: shape, min, max, avg, std (possibly along axes?)
# try SVD and/or QR to detect linear dependence
# K-Means Clustering (Asteroids, CERN?)
# Lasso (CERN?)
# n-dim FFT (CAMELS?)...


# "Asteroids": Asteroids of the Solar System
# Download the data set of the asteroids from the JPL Small Body Database from https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/
# and load the resulting csv file to Heat.


# ... to be completed ...

# "CAMELS": 1000 simulated universes on 128 x 128 x 128 grids
# Take a bunch of 1000 simulated universes from the CAMELS data set (8GB):
# ```
# wget https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/3D_grids/data/Nbody/Grids_Mtot_Nbody_Astrid_LH_128_z=0.0.npy ~/Grids_Mtot_Nbody_Astrid_LH_128_z=0.0.npy
# ```
# load them in NumPy, convert to PyTorch and Heat...

X_np = np.load("~/Grids_Mtot_Nbody_Astrid_LH_128_z=0.0.npy")

# ... to be completed ...

# "CERN": A particle physics data set from CERN
# Take a small part of the ATLAS Top Tagging Data Set from CERN (7.6GB, actually the "test"-part; the "train" part is much larger...)
# ```
# wget https://opendata.cern.ch/record/15013/files/test.h5 ~/test.h5
# ```
# and load it directly into Heat (watch out: the h5-file contains different data sets that need to be stacked...)

filename = "~/test.h5"
with h5py.File(filename, "r") as f:
    features = f.keys()
    arrays = [ht.load_hdf5(filename, feature, split=0) for feature in features]

# ... to be completed ...
