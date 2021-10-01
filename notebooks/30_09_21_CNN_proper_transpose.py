

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
import time
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as pl
import wandb
from wandb.keras import WandbCallback
from DataHandling.features.slices import load


#%%

root_logdir = os.path.join('/home/au643300/DataHandling/models', "my_logs")

def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_06_07-15_15_22

test=load('/home/au643300/DataHandling/data/processed/y_plus_15_test',repeat=(10))

train=load('/home/au643300/DataHandling/data/processed/y_plus_15_test',repeat=(10))

validation=load('/home/au643300/DataHandling/data/processed/y_plus_15_test',repeat=(10))

#%%

wandb.init(project="CNN_Baseline",group='...')

#Trying to make the model with keras functional api



weights=[128,256,256]
input=keras.layers.Input(shape=(256,256))
reshape=keras.layers.Reshape((256,256,1))(input)
batch=keras.layers.BatchNormalization(-1)(reshape)
cnn=keras.layers.Conv2D(64,5,activation='relu')(batch)
batch=keras.layers.BatchNormalization(-1)(cnn)
for weight in weights:
    cnn=keras.layers.Conv2D(weight,3,activation='relu')(batch)
    batch=keras.layers.BatchNormalization(-1)(cnn)
    
for weight in reversed(weights):
    cnn=keras.layers.Conv2DTranspose(weight,3,activation='relu')(batch)
    batch=keras.layers.BatchNormalization(-1)(cnn)



cnn=tf.keras.layers.Conv2DTranspose(64,5)(batch)
batch=keras.layers.BatchNormalization(-1)(cnn)
output=tf.keras.layers.Conv2DTranspose(1,1)(cnn)

model = keras.Model(inputs=input, outputs=output, name="CNN_baseline")

model.summary()


model.compile(loss="mean_squared_error", optimizer="Adam")


#%%
backup='/home/au643300/DataHandling/models/backup/'

str_time=time.strftime("%d-%m-%Y_%H%M")
backup_dir = os.path.join(backup, str_time)

#TODO move data to scratch first?
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
backup_cb=tf.keras.callbacks.ModelCheckpoint(backup_dir,save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=200,
restore_best_weights=True)
model.fit(x=test,epochs=100000,validation_data=validation,callbacks=[WandbCallback(),early_stopping_cb,backup_cb,tensorboard_cb])


