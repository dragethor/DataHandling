

#%%



from DataHandling.features import slices

from DataHandling import utility

from tensorflow import keras

import numpy as np

from DataHandling.features import slices
import os

import wandb

import sys
import shutil




os.environ['TF_ENABLE_ONEDNN_OPTS']='1'




slurm_arrary_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))

overwrite=False

def get_runs_wandb():
    api = wandb.Api()
    entity, project = "dragethor", "Thesis"  # set to your entity and project
    runs = api.runs(entity + "/" + project)
    summary_list, config_list, name_list = [], [], []
    
    
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        if run.state == "finished":
            summary_list.append(run.summary._json_dict)
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            config_list.append(
                {k: v for k,v in run.config.items()})
                #if not k.startswith('_')})
            # .name is the human-readable name of the run.
            name_list.append(run.name)
    return name_list, config_list


name_list, config_list = get_runs_wandb()






#%%


model_name=name_list[slurm_arrary_id]


config=config_list[slurm_arrary_id]


y_plus=config['y_plus']

var=config['variables']

target=[config['target']]

normalized=config['normalized']



#%%

# y_plus=15

# var=['u_vel']

# target=['tau_wall']

# normalized=True



# a=slices.slice_loc(y_plus,var,target,normalized)

#%%

model_path,output_path=utility.model_output_paths(model_name,y_plus,var,target,normalized)


print("Model "+ model_name,flush=True)



if not os.path.exists(output_path):

    os.makedirs(output_path)

elif os.path.exists(os.path.join(output_path,'targets.npz')) and overwrite==False:

    raise Exception("Data exists and overwrite is set to false. Exiting")

elif os.path.exists(os.path.join(output_path,'targets.npz')) and overwrite==True:

    print("deleting folder",flush=True)
    shutil.rmtree(output_path)




data=slices.load_validation(y_plus,var,target,normalized)


feature_list=[]

target_list=[]


for data_type in data:

    feature_list.append(data_type[0])

    target_list.append(data_type[1].numpy())




model=keras.models.load_model(model_path)


predctions=[]


predctions.append(model.predict(feature_list[0]))



predctions.append(model.predict(feature_list[1]))



predctions.append(model.predict(feature_list[2]))



predctions=[np.squeeze(x,axis=3) for x in predctions]


#%%


np.savez_compressed(os.path.join(output_path,"predictions"),train=predctions[0],val=predctions[1],test=predctions[2])

np.savez_compressed(os.path.join(output_path,"targets"),train=target_list[0],val=target_list[1],test=target_list[2])


print("Saved data",flush=True)