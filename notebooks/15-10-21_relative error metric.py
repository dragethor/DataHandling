
#%%
from DataHandling.features import slices
from DataHandling import utility
import xarray as xr
from DataHandling import plots
from tensorflow import keras
import os
import numpy as np
import pandas as pd

y_plus=15
repeat=5
shuffle=100
batch_size=10
activation='elu'
optimizer="adam"
loss='mean_squared_error'
patience=100
var=['u_vel']
target=['tau_wall']
normalize=False
dropout=False
model_name="lively-monkey-15"

#%%


model_path=os.path.join("/home/au643300/DataHandling/models/trained/",model_name)
data_path=slices.slice_loc(y_plus,var,target,normalize)+"/"
data_folder=os.path.basename(os.path.dirname(data_path))


data=slices.load_from_scratch(y_plus,var,target,normalized=normalize)
model=keras.models.load_model(model_path)


feature_list,target_list,names=utility.get_data(data)



predctions=[]
for features in feature_list:
    predctions.append(model.predict(features))

predctions=[np.squeeze(x,axis=3) for x in predctions]


#%%

error=pd.DataFrame(columns=['1-order mean','1-order fluctuating','2-order mean','2-order fluctuating'])





for i in range(3):

    mean_diff_2o=(np.mean(predctions[i][:,:,:]-target_list[i][:,:,:])**2)
    err_mean_2o=mean_diff_2o/(np.mean(target_list[i][:,:,:])**2)*100
    
    fluc_predict=predctions[i][:,:,:]-np.mean(predctions[i][:,:,:])
    fluc_target=target_list[i][:,:,:]-np.mean(target_list[i][:,:,:])
    
    err_fluct_2o= np.mean(fluc_predict-fluc_target)**2/(np.std(fluc_target)**2)
    
    
    error=error.append({'mean':err_mean,'fluctuating':err_fluct},ignore_index=True)





# %%
