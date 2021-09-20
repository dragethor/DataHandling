

#%%
import os
from operator import ne
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
from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pl
import wandb
from wandb.keras import WandbCallback
from DataHandling.features.slices import load


#%%


dataset=load("/home/au643300/DataHandling/data/processed/y_plus_15")





#%%

wandb.init(project="CNN_Guastoni")

#Trying to make the model with keras functional api



input=keras.layers.Input(shape=(256,256))
reshape=keras.layers.Reshape((256,256,1))(input)
batch=keras.layers.BatchNormalization(-1)(reshape)
cnn1=keras.layers.Conv2D(64,5,activation='relu')(batch)
batch1=keras.layers.BatchNormalization(-1)(cnn1)
cnn2=keras.layers.Conv2D(128,3,activation='relu')(batch1)
batch2=keras.layers.BatchNormalization(-1)(cnn2)
cnn3=keras.layers.Conv2D(256,3,activation='relu')(batch2)
batch3=keras.layers.BatchNormalization(-1)(cnn3)
cnn4=keras.layers.Conv2D(256,3,activation='relu')(batch3)
batch4=keras.layers.BatchNormalization(-1)(cnn4)
cnn5=keras.layers.Conv2D(128,3,activation='relu')(batch4)
output=tf.keras.layers.Conv2DTranspose(1,13)(cnn5)
#Evt kigge på at gøre billedet 128x128 og så upsamle det med depth to space?

model = keras.Model(inputs=input, outputs=output, name="CNN_Guastoni")

model.summary()




