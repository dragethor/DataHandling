

#%%



#%%

from operator import ne
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed,wait,fire_and_forget, LocalCluster
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
Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu #The friction velocity 
#y_plus=(y*utau/nu)

#Renaming the y value to y+
source=source.assign_coords(y=(source.y*u_tau/nu))
source=source.rename({'y':'y_plus'})


#%%

cluster=SLURMCluster(cores=8,
                     memory="50GB",
                     queue='q36',
                     walltime='0-02:00:00',
                     local_directory='/scratch/$SLURM_JOB_ID',
                     interface='ib0',
                     scheduler_options={'interface':'ib0'},
                     extra=['--resources mem=15',"--lifetime", "100m", "--lifetime-stagger", "4m"]
                    )

#%%
client=Client(cluster)

#%%
client
cluster.adapt(minimum_jobs=1,maximum_jobs=4)


#%%

#Taking out a single y+ value at y+=20
traning=source.sel(y_plus=20, method="nearest")

#The vertification data
results=source.sel(y_plus=0, method="nearest")

#taking out only the u vel of both
traning=traning['u_vel'].to_dataframe()
results=results['u_vel'].to_dataframe()


#TODO Der går noget galt her. Det er 100p pga det med at dataen er multidimensionel at jeg lige skal have fundet ud af...


#%%

results



