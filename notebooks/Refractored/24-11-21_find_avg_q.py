
#%%

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as pl
import wandb
from wandb.keras import WandbCallback
from DataHandling.features import slices
from DataHandling import utility
from DataHandling.models import models
from DataHandling import plots
import xarray as xr
import os



dt=xr.open_zarr('/home/au643300/DataHandling/data/interim/data.zarr')

Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity

#%%

vars=['pr1','pr0.71','pr0.2','pr0.025']

dt=dt[vars]

#%%

dt=dt.differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")



#%%
for i in list(dt.keys()):
    pr_number=float(i[2:])
    dt[i]=nu/(pr_number)*dt[i]

#%%
client,cluster =utility.slurm_q64(2)

dt=dt.mean()

#%%

dt=dt.compute()
