
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


client,cluster=utility.slurm_q64(2)

#%%

data=xr.open_zarr('/home/au643300/DataHandling/data/interim/data.zarr')


#%%

Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu

#Gætter på at rho=1

tau_wall=data['u_vel'].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")
tau_wall=nu*tau_wall
#%%
tau_wall_mean=tau_wall.mean().compute().values

#%%


u_tau_sim=np.sqrt(tau_wall_mean)

Re_tau_sim=u_tau_sim/nu
#Re_tau_sim=395.0162786