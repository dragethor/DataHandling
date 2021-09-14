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
import tensorflow.train as tft
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
cluster.adapt(minimum_jobs=0,maximum_jobs=6)


#%%

#Making a load function


def read_tfrecords(serial_data):
      """[reads tfrecords and unserializeses them]

      Args:
          serial_data ([TFrecord]): [Tfrecord that needs to be unserialzed]

      Returns:
          [tuple]: [A tuple of u_vel and tau_wall]
      """
      format = {
      "u_vel": tf.io.FixedLenFeature([], tf.string, default_value=""),
      "tau_wall": tf.io.FixedLenFeature([], tf.string, default_value="")
      }

      features=tf.io.parse_single_example(serial_data, format)

      u_vel=tf.io.parse_tensor(features['u_vel'],tf.float64)
      tau_wall=tf.io.parse_tensor(features['tau_wall'],tf.float64)
      return (u_vel, tau_wall)




data_loc="/home/au643300/DataHandling/data/processed/y_plus_15"

dataset = tf.data.TFRecordDataset([data_loc],compression_type='GZIP')
dataset=dataset.range(10).batch(20).shuffle(buffer_size=100).map(read_tfrecords)


#NU ER ALT FUCKING KLART TIL AT LOADE IN I FUCKING TENSORFLOW
#Evt samle det hele som på side 410 i bogen?



#%%

for serialized_example in dataset:
      parsed_example = tf.io.parse_example(serialized_example,format)







#Trying to get the file out again
#%%








#%%


