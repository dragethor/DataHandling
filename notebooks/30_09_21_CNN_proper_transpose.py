

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
#%%


#scratch slurm save spot

scratch=os.path.join('/scratch/', os.environ['SLURM_JOB_ID'])
data_loc='/home/au643300/DataHandling/data/processed/y_plus_15_test'




#TODO lavet det her færdigt og lave et eller andet smart så jeg ikke kommer til kalde det samme data 3 gange...
#TODO ryddet op i det her script så det er klar til flere runs...
#TODO evt gjort med at ændre load til at finde en mappe
#TODO hvordan skal jeg gøre når jeg skal loade andet/mere data med mine filer? Evt noget med en dict






#%%


y_plus=15
repeat=15
shuffle=100
batch_size=10
activation='relu'
optimizer="adam"
loss='mean_squared_error'
patience=200

wandb.init(project="CNN_Baseline",sync_tensorboard=True)
config=wandb.config
config.y_plus=15
config.repeat=repeat
config.shuffle=shuffle
config.batch_size=batch_size
config.activation=activation
config.optimizer=optimizer
config.loss=loss
config.patience=patience


utility.load_from_scratch(y_plus,repeat=repeat,shuffle_size=shuffle,batch_s=batch_size)



#Trying to make the model with keras functional api



weights=[128,256,256]
input=keras.layers.Input(shape=(256,256))
reshape=keras.layers.Reshape((256,256,1))(input)
batch=keras.layers.BatchNormalization(-1)(reshape)
cnn=keras.layers.Conv2D(64,5,activation=activation)(batch)
batch=keras.layers.BatchNormalization(-1)(cnn)
for weight in weights:
    cnn=keras.layers.Conv2D(weight,3,activation=activation)(batch)
    batch=keras.layers.BatchNormalization(-1)(cnn)
    
for weight in reversed(weights):
    cnn=keras.layers.Conv2DTranspose(weight,3,activation=activation)(batch)
    batch=keras.layers.BatchNormalization(-1)(cnn)



cnn=tf.keras.layers.Conv2DTranspose(64,5)(batch)
batch=keras.layers.BatchNormalization(-1)(cnn)
output=tf.keras.layers.Conv2DTranspose(1,1)(cnn)

model = keras.Model(inputs=input, outputs=output, name="CNN_baseline")

model.summary()


model.compile(loss=loss, optimizer=optimizer)


#%%

backup_dir , log_dir= utility.get_run_dir(wandb.run.name)




tensorboard_cb = keras.callbacks.TensorBoard(log_dir)
backup_cb=tf.keras.callbacks.ModelCheckpoint(backup_dir,save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=200,
restore_best_weights=True)
model.fit(x=test,epochs=100000,validation_data=validation,callbacks=[WandbCallback(),early_stopping_cb,backup_cb,tensorboard_cb])


