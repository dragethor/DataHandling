





""" 
Et script som tester alle modeller fra wandb ved andre Pr tal hvis de er lavet til at forudsige Pr.
Den skal bruges sammen med array_run batch scriptet
"""



#%%

from DataHandling.features import slices
from DataHandling import utility
from tensorflow import keras
from DataHandling.models import predict
import os


name_list, config_list = utility.get_runs_wandb()

overwrite=False


#Bruges for at køre forskellige modeller samtidigt via SLURM arrays. Bruges sammen med array_run batch filen
slurm_arrary_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))



name=name_list[slurm_arrary_id]
config=config_list[slurm_arrary_id]

#name=name_list[0]
#config=config_list[0]

print("This is " + name,flush=True)




#%%

pr_numbers=["pr0.2",'pr0.025',"pr1"]

if config['target']=="pr0.71_flux":
    
    #first load the model
    y_plus=int(config['y_plus'])
    vars=sorted(config['variables'])
    target=[config['target']]
    normalize=config['normalized']
    model_path, _ =utility.model_output_paths(name,y_plus,vars,target,normalize)
    model=keras.models.load_model(model_path)
    
    #Predict for standart config as well
    #print("testing at " + str(vars) +"Target " +str(target), flush=True)
    #predict.predict(name,overwrite,model,y_plus,vars,target,normalize)


    #now change the target and vars to other pr
    for pr_number in pr_numbers:
        target=[pr_number+"_flux"]
        if vars[0][:2]=="pr":
            vars[0]=pr_number
            
            for layer in model.layers:
                if layer.name[0:2]=="pr":
                    layer._name=pr_number
        print("testing at " + str(vars) +"Target " +str(target), flush=True)
        #predict at the other pr numbers and save the data
        predict.predict(name,overwrite,model,y_plus,vars,target,normalize)









