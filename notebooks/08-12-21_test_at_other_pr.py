
#%%

from DataHandling.features import slices
from DataHandling import utility
from tensorflow import keras
from DataHandling.models import predict
import os


name_list, config_list = utility.get_runs_wandb()

overwrite=False

slurm_arrary_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))

#%%

name=name_list[slurm_arrary_id]
config=config_list[slurm_arrary_id]

pr_numbers=['pr0.025',"pr0.2","pr1"]

if config['target']=="pr0.71_flux":
    
    #first load the model
    y_plus=int(config['y_plus'])
    vars=config['variables']
    target=config['target']
    normalize=config['normalized']
    model_path, _ =utility.model_output_paths(name,y_plus,vars,target,normalize)
    model=keras.models.load_model(model_path)

    #now change the target and vars to other pr
    for pr_number in pr_numbers:
        target=[pr_number+"_flux"]
        if "pr0.71" in vars:
            pr_var_index=vars.index("pr0.71")
            vars[pr_var_index]=pr_number

        #predict at the other pr numbers and save the data
        predict.predict(name,overwrite,model,y_plus,vars,target,normalize)










