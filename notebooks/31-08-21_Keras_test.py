

#%%



#%%

from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed,wait,fire_and_forget
import glob
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import importlib
import dask
import zarr
import time

from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pl


#%%

sns.set_theme()
dask.config.set({"distributed.comm.timeouts.tcp": "50s"})
dask.config.set({'distributed.comm.retry.count':3})
dask.config.set({'distributed.comm.timeouts.connect':'25s'})
dask.config.set({"distributed.worker.use-file-locking":False})

#%%

source=xr.open_zarr("/home/au643300/NOBACKUP/data/interim/data.zarr/")

