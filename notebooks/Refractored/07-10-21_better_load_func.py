
#%%

import os
from posixpath import join
import xarray as xr
import numpy as np
import dask
import tensorflow as tf
from DataHandling import utility
from DataHandling import features
from DataHandling.features import slices
import shutil
import json





#%%

df=xr.open_zarr("/home/au643300/DataHandling/data/interim/data.zarr")

y_plus=15
var=['u_vel']
target=['pr1']

#%%

slices.save_tf(y_plus,var,target,df)


#%%
save_loc=os.path.join("/home/au643300/DataHandling/data/processed",'y_plus_'+str(y_plus))+"_var_"+str(len(var))

data=slices.load_from_scratch(15,var)



#%%
sent1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
sent2 = np.array([20, 25, 35, 40, 600, 30, 20, 30], dtype=np.int64)
labels = np.array([40, 30, 20, 10, 80, 70, 50, 60], dtype=np.int64)
labels = np.reshape(labels, (8))

dataset = tf.data.Dataset.from_tensor_slices(({"input_1": sent1, "input_2": sent2}, labels))

#%%

for i in data[0].take(1):
    a=i

#%%
#slices.save_tf(y_plus,var,df)


# %%
