
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
var=['u_vel',"pr0.71"]
target=['pr0.71_flux']
normalize=False
dropout=False
model_name="hopeful-lion-18"


#%%

model_path,output_path=utility.model_output_paths(model_name,y_plus,var,target,normalize)

feature_list,target_list,predctions,names=utility.get_data(model_name,y_plus,var,target,normalize)


error_fluc,err=plots.error(target_list,names,predctions,output_path)

plot=plots.heatmaps(target_list,names,predctions,output_path,model_path,target)

pdf_plt=plots.pdf_plots(error_fluc,names,output_path)
