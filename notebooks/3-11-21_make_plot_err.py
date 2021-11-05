
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
from DataHandling import plots
import importlib



y_plus=15
repeat=5
shuffle=100
batch_size=10
activation='elu'
optimizer="adam"
loss='mean_squared_error'
patience=100
var=['u_vel']
target=['tau_wall']
normalize=True
dropout=False
model_name="peach-rain-14"


#%%

model_path,output_path=utility.model_output_paths(model_name,y_plus,var,target,normalize)

feature_list,target_list,predctions,names=utility.get_data(model_name,y_plus,var,target,normalize)


#%%
#importlib.reload(plots)
error_fluc,err=plots.error(target_list,names,predctions,output_path)

plot=plots.heatmaps(target_list,names,predctions,output_path,model_path,target)



#%%




#%%

# f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
# ax=sns.violinplot(data=error_fluc[2]['Local Mean Error'],ax=ax)
# ax2=sns.violinplot(data=error_fluc[2]['Local Mean Error'],ax=ax2)

# ax.set_ylim(60, error_fluc[2]['Local Mean Error'].max())  # outliers only
# ax2.set_ylim(-10, 60)

# ax.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax.xaxis.tick_top()
# ax.tick_params(labeltop=False)  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()

# d = .015
# kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# plt.show()



#%%
cm =1/2.54

fig, (ax, ax2) = plt.subplots(2, 1,figsize=(20*cm,20*cm),dpi=200)
sns.boxplot(data=error_fluc[2][['MAE fluct','MSE fluct']],showfliers = False,orient='h',ax=ax)
sns.boxplot(data=error_fluc[2][['MAE local','MSE local']],showfliers = False,orient='h',ax=ax2)
fig.savefig(os.path.join(output_path,'boxplot.pdf'),bbox_inches='tight',format='pdf')


#%%
plt.figure(figsize=(20*cm,10*cm),dpi=200)
sns.histplot(data=error_fluc[2],stat='density',kde=True,log_scale=True)
plt.xlim(1*10**(-2),1*10**(4))
plt.savefig(os.path.join(output_path,'PDF.pdf'),bbox_inches='tight',format='pdf')
