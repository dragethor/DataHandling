
#%%
from DataHandling.features import slices
from DataHandling import utility
from tensorflow import keras
import os
import numpy as np
import pandas as pd
import shutil

# %%


def error(target_list,names,predctions,output_path,model_path):


    from DataHandling.features import slices
    from DataHandling import utility
    from tensorflow import keras
    import os
    import numpy as np
    import pandas as pd
    import shutil
    

    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        print('deleting old version')
        shutil.rmtree(output_path)           
        os.makedirs(output_path)

 
    
    
    error=pd.DataFrame(columns=['Global Mean Error','Local Mean Error','Fluctuating Error global','Fluctuating Error Local','Fluctuating Error Std'])
    error_fluc_list=[]
    
    for i in range(3):
        error_fluct=pd.DataFrame()
        
        
        fluc_predict=predctions[i][:,:,:]-np.mean(predctions[i][:,:,:])
        fluc_target=target_list[i][:,:,:]-np.mean(target_list[i][:,:,:])
        
        err_mean_global=(np.mean(predctions[i][:,:,:])-np.mean(target_list[i][:,:,:]))/(np.mean(target_list[i][:,:,:]))*100
        err_mean_local_sqrt=np.sqrt((np.mean((predctions[i][:,:,:]-target_list[i][:,:,:])**2))/np.mean(target_list[i][:,:,:])**2)*100

        

        err_fluc_sigma=(np.std(fluc_predict)-np.std(fluc_target))/(np.std(fluc_target))*100
        err_fluc_loc_sqrt=np.sqrt((np.mean(fluc_predict-fluc_target)**2)/np.std(fluc_target)**2)*100

        err_local_non_mean_sqrt=np.sqrt(((predctions[i][:,:,:]-target_list[i][:,:,:])**2)/np.mean(target_list[i][:,:,:])**2)*100
        err_fluc_local_non_mean_sqrt=np.sqrt(((fluc_predict-fluc_target)**2)/np.std(fluc_target)**2)*100
        
        
        error_fluct['Local Mean Error']=err_local_non_mean_sqrt.flatten()
        error_fluct['Local fluct Error']=err_fluc_local_non_mean_sqrt.flatten()
        
        error_fluct.to_csv(os.path.join(output_path,'Error_fluct_'+names[i]+'.csv'))
        error_fluc_list.append(error_fluct)
        
        error=error.append({'Global Mean Error':err_mean_global,'Local Mean Error':err_mean_local_sqrt,'Fluctuating Error Local':err_fluc_loc_sqrt,'Fluctuating Error Std':err_fluc_sigma},ignore_index=True)

    error.index=names

    error.to_csv(os.path.join(output_path,'Mean_error.csv'))


    return error_fluc_list, error



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
normalize=False
dropout=False
model_name="lively-monkey-15"

feature_list,target_list,names=utility.get_data(y_plus,var,target,normalize)

#%%

fluc_list,err=error(feature_list,target_list,names,model_name,var,target,y_plus,normalize)


#%%

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


