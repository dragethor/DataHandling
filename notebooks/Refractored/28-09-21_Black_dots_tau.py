

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
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as pl
from wandb.keras import WandbCallback
from DataHandling.features.slices import load
from DataHandling.data_raw.make_dataset import readDNSdata





#%%

data=xr.open_zarr('/home/au643300/DataHandling/data/interim/data.zarr')

u_vel=data.isel(time=0)['u_vel'].values


y_ax=data['y'].values
x_ax=data['x'].values
z_ax=data['z'].values
tau=data['u_vel'].isel(time=0).differentiate('y')
tau=tau.values


#%%

tau_numpy=np.diff(u_vel,axis=1)

#%%


f,(ax1,ax2)=plt.subplots(1,2,dpi=300)

sns.heatmap(np.transpose(tau_numpy[:,-1,:]),ax=ax1,square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(tau[:,-1,:]),ax=ax2,square=True,xticklabels=False,yticklabels=False)




#%%
f,(ax3)=plt.subplots(2,2,dpi=300)
sns.heatmap(np.transpose(u_vel[:,-1,:]),ax=ax3[0,0],square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(u_vel[:,-2,:]),ax=ax3[0,1],square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(u_vel[:,-3,:]),ax=ax3[1,0],square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(u_vel[:,-4,:]),ax=ax3[1,1],square=True,xticklabels=False,yticklabels=False)
f.suptitle('Data from xarray')


#%%
#Trying to read data directly from .u file

quantities, _, xf, yf, zf, length, _, _ = readDNSdata('/home/au643300/NOBACKUP/run/21_02_24/field.0493.u')

#%%

u_raw= quantities[0]

f,(ax4)=plt.subplots(2,3,dpi=300)
sns.heatmap(np.transpose(u_raw[:,-1,:]),ax=ax4[0,0],square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(u_raw[:,-3,:]),ax=ax4[0,1],square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(u_raw[:,-5,:]),ax=ax4[0,2],square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(u_raw[:,-7,:]),ax=ax4[1,0],square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(u_raw[:,-9,:]),ax=ax4[1,1],square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(u_raw[:,-12,:]),ax=ax4[1,2],square=True,xticklabels=False,yticklabels=False)
f.suptitle('Data from raw files at different y values')


#%%
f,(ax5)=plt.subplots(2,3,dpi=300)
sns.heatmap(np.transpose(u_raw[:,:,-1]),ax=ax5[0,0],square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(u_raw[:,:,-10]),ax=ax5[0,1],square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(u_raw[:,:,-20]),ax=ax5[0,2],square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(u_raw[:,:,-30]),ax=ax5[1,0],square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(u_raw[:,:,-40]),ax=ax5[1,1],square=True,xticklabels=False,yticklabels=False)
sns.heatmap(np.transpose(u_raw[:,:,-50]),ax=ax5[1,2],square=True,xticklabels=False,yticklabels=False)
f.suptitle('Data from raw files at different z values')


#%%
Z,X = np.meshgrid(z_ax,x_ax)
plt.figure(dpi=400)
plt.contour(Z,X, np.transpose(u_raw[:,0,:]) )