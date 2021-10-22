



#%%
import os
from operator import ne
import re
from dask.base import optimize
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed,wait,fire_and_forget, LocalCluster
import glob
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dask
import zarr
import time
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as pl
import wandb
from wandb.keras import WandbCallback
from DataHandling.features import slices
import shutil
from DataHandling import utility
from DataHandling.models import models


#%%


slice_array=xr.open_zarr("/home/au643300/DataHandling/data/interim/data.zarr")

    
#%%

target=['tau_wall']

normalized=True

var=['u_vel','tau_wall']

y_plus=15

#%%




if target[0]=='tau_wall':
    target_slice=slice_array['u_vel'].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")
    if normalized==True:
        target_slice=(target_slice-target_slice.mean(dim=('time','x','z')))/(target_slice.std(dim=('time','x','z')))
else:
    target_slice=slice_array[target[0]].sel(y=utility.y_plus_to_y(0),method="nearest")
    if normalized==True:
        target_slice=(target_slice-target_slice.mean(dim=('time','x','z')))/(target_slice.std(dim=('time','x','z')))


slice_array=slice_array.sel(y=utility.y_plus_to_y(y_plus), method="nearest")
if normalized==True:
    slice_array=(slice_array-slice_array.mean(dim=('time','x','z')))/(slice_array.std(dim=('time','x','z')))

slice_array[target[0]]=target_slice
slice_array=slice_array[var]

slice_array=dask.compute(slice_array,retries=5)[0]
