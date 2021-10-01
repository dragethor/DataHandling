
#%%
import os
from operator import ne
import re
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed,wait,fire_and_forget, LocalCluster
import glob
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.util.nest import flatten
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
from DataHandling.features.slices import load
import seaborn as sns


data=xr.open_zarr('/home/au643300/DataHandling/data/interim/data.zarr')


Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu

#converts between y_plus and y
y_func= lambda y_plus : y_plus*nu/u_tau


#%%

u_test=data['u_vel'].isel(time=1).sel(y=y_func(15), method="nearest")
u_test=u_test.values

tau_wall=data['u_vel'].differentiate('y').isel(time=1).isel(y=-1)
tau_wall=tau_wall.values

plt.figure()

p1=sns.heatmap(np.transpose(u_test),square=True,xticklabels=False,yticklabels=False)
plt.title('test of u_vel to check for spots')

plt.figure()

p1=sns.heatmap(np.transpose(tau_wall),square=True,xticklabels=False,yticklabels=False)
plt.title('test of tau_w to check for spots')



#%%
model=keras.models.load_model('/home/au643300/DataHandling/models/trained/CNN_like_gustanoi_Conv2DTranspose.h5')

test=load('/home/au643300/DataHandling/data/processed/y_plus_15_test',repeat=(1))

for item in test.take(1):
    u_vel_tensorflow=item[0]
    tau_mark=item[1]

tau_p=model.predict(u_vel_tensorflow)





#%%
plt.figure()

p1=sns.heatmap(np.transpose(tau_mark.numpy()[1,:,:]),square=True,xticklabels=False,yticklabels=False)
plt.title('Ground Truth')

plt.figure()
p2=sns.heatmap(np.transpose(np.squeeze(tau_p[1,:,:,:],axis=2)),square=True,xticklabels=False,yticklabels=False)
plt.title('Prediction')

