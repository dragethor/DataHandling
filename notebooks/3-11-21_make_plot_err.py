
#%%

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as pl
import wandb
from wandb.keras import WandbCallback
from DataHandling.features import slices
from DataHandling import utility
from DataHandling.models import models
from DataHandling import plots
import xarray as xr
import os

os.environ['TF_ENABLE_ONEDNN_OPTS']='1'

 
y_plus=15
activation='elu'
optimizer="adam"
loss='mean_squared_error'
var=['u_vel']
target=['tau_wall']
target_type='stress'
normalize=False
model_names="lively-monkey-15"


#for name in model_names:

#%%
model_path, output_path =utility.model_output_paths(model_names,y_plus,var,target,normalize)
#%%

feature_list, target_list, predctions, names= utility.get_data(model_names,y_plus,var,target,normalize)

#%%

error_fluc,err= plots.error(target_list,target_type,names,predctions,output_path)
#%%
plots.heatmaps(target_list,names,predctions,output_path,model_path,target)
#%% 
plots.pdf_plots(error_fluc,names,output_path)

# %%
