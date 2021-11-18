
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



y_plus=15
activation='elu'
optimizer="adam"
loss='mean_squared_error'
var=['u_vel',"pr0.71"]
target=['pr0.71_flux']
target_type='flux'
normalize=False
model_names=["hopeful-lion-18"]




for name in model_names:

    model_path, output_path =utility.model_output_paths(name,y_plus,var,target,normalize)

    feature_list, target_list, predctions, names= utility.get_data(name,y_plus,var,target,normalize)




    error_fluc,err=plots.error(target_list,target_type,names,predctions,output_path)

    plots.heatmaps(target_list,names,predctions,output_path,model_path,target)

    plots.pdf_plots(error_fluc,names,output_path)

# %%
