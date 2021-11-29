

#%%
import os
import numpy as np
from glob import glob
from DataHandling import utility
from DataHandling import plots


output_path="/home/au643300/DataHandling/models/output"
model_names=os.listdir(output_path)

#%%

for model in model_names:

    full_dir=os.path.join(output_path,model)
    subdirs=os.listdir(full_dir)


    #Skal bruge et forloop her

    for dir in subdirs:
        dir_split=dir.split("-")

        y_plus=int(dir_split[0][-2:])

        index_vars_s=dir_split.index("VARS1")
        index_target=dir_split.index("TARGETS")


        var=dir_split[index_vars_s+1:index_target]
        target=dir_split[index_target+1:]
        if "normalized" not in dir_split:
            normalize=False
        else:
            normalize=True
            target=target[:-1]

        if target[0][-5:] =='_flux':
            target_type="flux"
        elif target[0]=='tau_wall':
            target_type="stress"

        model_path, _ =utility.model_output_paths(model,y_plus,var,target,normalize)


        full_subpath=os.path.join(full_dir,dir)
        prediction_path=os.path.join(full_subpath,'predictions.npz')
        target_path=os.path.join(full_subpath,'targets.npz')

        if os.path.exists(prediction_path) and os.path.exists(target_path):
            pred=np.load(prediction_path)
            targ=np.load(target_path)
            target_list=[targ["train"],targ["val"],targ["test"]]
            predctions=[pred["train"],pred["val"],pred["test"]]

            names=["train","validation","test"]

            plots.heatmaps(target_list,names,predctions,output_path,model_path,target)
            error_fluc,err= plots.error(target_list,target_type,names,predctions,output_path)
            plots.pdf_plots(error_fluc,names,output_path)

#Skal have sat en overwrite statement ind, og få den til at kigge om filerne allerede findes, og springe over hvis de gør