#%%
import os
from operator import ne
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed,wait,fire_and_forget, LocalCluster
import glob
from tensorflow.core.example.feature_pb2 import Features
from tensorflow.python.util.nest import flatten
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

AUTOTUNE = tf.data.experimental.AUTOTUNE

#%%

source=xr.open_zarr("/home/au643300/NOBACKUP/data/interim/data.zarr/")


cluster=SLURMCluster(cores=8,
                     memory="50GB",
                     queue='q64',
                     walltime='0-00:30:00',
                     local_directory='/scratch/$SLURM_JOB_ID',
                     interface='ib0',
                     scheduler_options={'interface':'ib0'},
                     extra=['--resources mem=15',"--lifetime", "20m"]
                    )
#"--lifetime-stagger", "4m"



#%%
client=Client(cluster)
cluster.adapt(minimum_jobs=0,maximum_jobs=4)





#%%

y_plus=15


Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu

#converts between y_plus and y
y_func= lambda y_plus : y_plus*nu/u_tau


slice=slice.assign(tau_wall=slice['u_vel'].differentiate('y').isel(y=-1))
slice=source.sel(y=y_func(15), method="nearest")






#results=source=source.differentiate('y')
#results=results.isel(y=-1)

#Så først tages der her og slices sådan at man kun har de y_værdier man vil have, og derudover også tau ved y=-1


#%%
slice=slice.compute()



#%%
#Test hvor jeg kun gemmer u og tau


#TODO Mangler noget her. Fra https://www.tensorflow.org/api_docs/python/tf/io/serialize_tensor ved "Serialize the data using"

def serialize(u_vel,tau_wall):
      feature = {
      'u_vel': tf.io.serialize_tensor(u_vel),
      'tau_wall': tf.io.serialize_tensor(tau_wall),
      }
      serialized=tf.train.Example.Features(feature=feature)
      return serialized


u_vel=slice['u_vel'].data
tau_wall=slice['tau_w'].data

with tf.io.TFRecordWriter(filename) as writer:
      for i in range(len(slice.time)):
            test=serialize(u_vel,tau_wall)
            writer.write(test)






# %%
#Det der kommer til at ske er at jeg udtrækker ved et antal y+ værdier og gemmer dem som tfrecords






feature = {
      'u_vel': _int64_feature(image_shape[0]),
      'v_vel': _int64_feature(image_shape[1]),
      'w_vel': _int64_feature(image_shape[2]),
      'tau_wall': _int64_feature(label),
      'pr0.025': _bytes_feature(image_string),
      'pr0.2': _bytes_feature(image_string),
      'pr0.71': _bytes_feature(image_string),
      'pr0.1': _bytes_feature(image_string),
      'y_plus' aaa,




def slices_to_records(y_plus,):

