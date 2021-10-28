

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



#%%


y_plus=15
repeat=5
shuffle=100
batch_size=10
activation='elu'
optimizer="adam"
loss='mean_squared_error'
patience=50
var=['u_vel']
target=['tau_wall']
normalized=True
dropout=False

data=slices.load_from_scratch(y_plus,var,target,normalized,repeat=repeat,shuffle_size=shuffle,batch_s=batch_size)

train=data[0]
validation=data[1]




#%%
#Wandb stuff
wandb.init(project="Thesis",notes="Test of baseline network with normalized data")
config=wandb.config
config.y_plus=y_plus
config.repeat=repeat
config.shuffle=shuffle
config.batch_size=batch_size
config.activation=activation
config.optimizer=optimizer
config.loss=loss
config.patience=patience
config.variables=var
config.target=target[0]
config.dropout=dropout
config.normalized=normalized


model=models.baseline_cnn(activation)

model.summary()


model.compile(loss=loss, optimizer=optimizer)



logdir, backupdir= utility.get_run_dir(wandb.run.name)




tensorboard_cb = keras.callbacks.TensorBoard(logdir)
backup_cb=tf.keras.callbacks.ModelCheckpoint(backupdir,save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience,
restore_best_weights=True)
model.fit(x=train,epochs=100000,validation_data=validation,callbacks=[WandbCallback(),early_stopping_cb,backup_cb,tensorboard_cb])

model.save(os.path.join("/home/au643300/DataHandling/models/trained",wandb.run.name))

