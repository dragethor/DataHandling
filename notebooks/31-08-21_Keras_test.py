


#%%
import os
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
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pl
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
                     walltime='0-00:30:00',
                     local_directory='/scratch/$SLURM_JOB_ID',
                     interface='ib0',
                     scheduler_options={'interface':'ib0'},
                     extra=['--resources mem=15',"--lifetime", "20m"]
                    )
#"--lifetime-stagger", "4m"
#%%
client=Client(cluster)

#%%
client
cluster.adapt(minimum_jobs=0,maximum_jobs=4)


#%%

#Taking out a single y+ value at y+=20
traning=source.sel(y_plus=20, method="nearest")

#The vertification data
results=source.isel(y_plus=-2)



#%%

traning_u=traning['u_vel']
results_u=results['u_vel']


traning_u=traning_u.compute()
results_u=results_u.compute()



traning_un=traning_u.data
results_un=results_u.data

traning_un=np.expand_dims(traning_un,0)
traning_un=np.swapaxes(traning_un,0,1)
traning_un=np.moveaxis(traning_un,1,-1)



results_un=np.expand_dims(results_un,0)
results_un=np.swapaxes(results_un,0,1)
results_un=np.moveaxis(results_un,1,-1)


#Så nu har jeg alt min data på numpy form. Konveterer det til tensorflow tensor




#%%

tensor=tf.data.Dataset.from_tensor_slices((traning_un,results_un))

tensor=tensor.shuffle(10)

#%%
#Splitting data into test,val, train

train_size=int(0.7*681)
val_size=int(0.15*681)
test_size=int(0.15*681)



train=tensor.take(train_size)
test=tensor.skip(train_size)
val=test.skip(val_size)
test=test.take(test_size)


#%%


#%%

#Prøver at bruge CN fra CNN guide fil

num_filters =8
filter_size=3
pool_size=2

#TODO her går noget galt. den siger at der mangler en dim. Tror det handler om den måde at tensor_from_slices wirker på.


model = keras.models.Sequential([
    layers.BatchNormalization(-1,input_shape=(256,256,1)),
    layers.Conv2D(num_filters,filter_size),
    layers.MaxPooling2D(pool_size=pool_size),
    layers.Flatten(),
    layers.Dense(256,activation='relu')
])





#%%

model.summary()


#%%

model.compile(
  'adam',
  loss='mse',
  metrics=['mse'],
)

#%%

a=list(train.as_numpy_iterator())[0][0]


#%%


model.fit(
    x=train,
    epochs=2,
    validation_data=val,
    use_multiprocessing=True
)
