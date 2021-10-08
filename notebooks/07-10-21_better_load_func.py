
#%%
import os
import xarray as xr
import numpy as np
import dask
import tensorflow as tf
from DataHandling import utility
from DataHandling.features import slices
import shutil
import json





y_plus=15
var=['u_vel']

dt=xr.open_zarr("/home/au643300/DataHandling/data/interim/data.zarr")

slices.save_tf(15,['u_vel'],dt)



# %%
