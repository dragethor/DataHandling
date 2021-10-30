
#%%
from DataHandling.features import slices
from DataHandling import utility
import xarray as xr
from DataHandling import plots
from tensorflow import keras
import os
import numpy as np
import pandas as pd


# %%


def error(model_name,var,target,y_plus,normalize):
    """Find the first and second order error between the prediction and the target for flucturations and predictions.

    Args:
        model_name (Wandb Name): Name given to run by wandb
        var (list): list of variables without target
        target (list): list of target
        y_plus (int): y_plus value used for planes
        normalize (bool): if the data is normalized or not

    Returns:
        error (DataFrame): A Dataframe containing the first and second order error for both flucturations and mean
    """
    output_path='/home/au643300/DataHandling/models/output'
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
    
    
    
    error=pd.DataFrame(columns=['1-order mean','1-order fluctuating','2-order mean','2-order fluctuating'])


    for i in range(3):

        fluc_predict=predctions[i][:,:,:]-np.mean(predctions[i][:,:,:])
        fluc_target=target_list[i][:,:,:]-np.mean(target_list[i][:,:,:])
        
        err_mean_global=(np.mean(predctions[i][:,:,:])-np.mean(target_list[i][:,:,:]))/(np.mean(target_list[i][:,:,:]))
        err_mean_local_sqrt=np.sqrt((np.mean((predctions[i][:,:,:]-target_list[i][:,:,:])**2))/np.mean(target_list[i][:,:,:])**2)


        err_fluc_sigma=(np.std(fluc_predict)-np.std(fluc_target))/(np.std(fluc_target))
        err_fluc_loc_sqrt=np.sqrt((np.mean(fluc_predict-fluc_target)**2))/np.std(fluc_target)**2)
        err_fluc_global=(np.mean(fluc_predict)-np.mean(fluc_target))/(np.mean(fluc_target))


        error=error.append({'1-order mean':err_mean_1o,'1-order fluctuating':err_fluct_1o,'2-order mean':err_mean_2o,'2-order fluctuating':err_fluct_2o},ignore_index=True)

    error.index=names

    error.to_csv(os.path.join(output_path,model_name+'_'+data_folder))


    return error

#%%


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
normalize=True
dropout=False
model_name="peach-rain-14"




model_path=os.path.join("/home/au643300/DataHandling/models/trained/",model_name)
data_path=slices.slice_loc(y_plus,var,target,normalize)+"/"
data_folder=os.path.basename(os.path.dirname(data_path))


data=slices.load_from_scratch(y_plus,var,target,normalized=normalize)
model=keras.models.load_model(model_path)


feature_list,target_list,names=utility.get_data(data)

#%%

predctions=[]
for features in feature_list:
    predctions.append(model.predict(features))

predctions=[np.squeeze(x,axis=3) for x in predctions]


#%%

error=pd.DataFrame(columns=['1-order mean','1-order fluctuating','2-order mean','2-order fluctuating'])


for i in range(3):

    fluc_predict=predctions[i][:,:,:]-np.mean(predctions[i][:,:,:])
    fluc_target=target_list[i][:,:,:]-np.mean(target_list[i][:,:,:])
    
    mean_diff_1o=(np.mean(predctions[i][:,:,:]-target_list[i][:,:,:]))
    err_mean_1o=mean_diff_1o/(np.mean(target_list[i][:,:,:]))*100
    err_fluct_1o= np.mean(fluc_predict-fluc_target)/(np.std(fluc_target))
    
    mean_diff_2o=(np.mean(predctions[i][:,:,:]-target_list[i][:,:,:])**2)
    err_mean_2o=mean_diff_2o/(np.mean(target_list[i][:,:,:])**2)*100
    
    err_fluct_2o= np.mean(fluc_predict-fluc_target)**2/(np.std(fluc_target)**2)

    error=error.append({'1-order mean':err_mean_1o,'1-order fluctuating':err_fluct_1o,'2-order mean':err_mean_2o,'2-order fluctuating':err_fluct_2o},ignore_index=True)


