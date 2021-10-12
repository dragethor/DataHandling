

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

client,cluster = utility.slurm_q64(2)


#%%

df=xr.open_zarr("/home/au643300/DataHandling/data/interim/data.zarr")


df=df[['u_vel','v_vel']]

b=df.mean()


c=(df-b)/(df.max()-df.min())