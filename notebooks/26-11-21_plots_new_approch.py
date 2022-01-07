

#%%
import os
import matplotlib
import importlib
matplotlib.use('Agg')


import numpy as np
from glob import glob
from DataHandling import utility
from DataHandling import plots
from zipfile import BadZipfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt



overwrite=False
overwrite_pics=False
overwrite_pdf=True

path_of_output="/home/au643300/DataHandling/models/output"
name_list, _ = utility.get_runs_wandb()

slurm_arrary_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))


#%%


model=name_list[slurm_arrary_id]
#importlib.reload(plots)
#model=name_list[-4]

full_dir=os.path.join(path_of_output,model)
subdirs=os.listdir(full_dir)

print('This is model ' + model,flush=True)

for dir in subdirs:
    dir_split = dir.split("-")

    y_plus = int(dir_split[0][-2:])

    index_vars_s = dir_split.index("VARS")
    index_target = dir_split.index("TARGETS")

    var = dir_split[index_vars_s+1:index_target]
    target = dir_split[index_target+1:]
    if "normalized" not in dir_split:
        normalize = False
    else:
        normalize = True
        target = target[:-1]

    if target[0][-5:] == '_flux':
        target_type = "flux"
    elif target[0] == 'tau_wall':
        target_type = "stress"

    model_path, output_path =utility.model_output_paths(model,y_plus,var,target,normalize)


    prediction_path=os.path.join(output_path,'predictions.npz')
    target_path=os.path.join(output_path,'targets.npz')

    if os.path.exists(prediction_path) and os.path.exists(target_path):
        try:
            scratch=os.path.join('/scratch/', os.environ['SLURM_JOB_ID'])
            prediction_scratch=os.path.join(scratch,'predictions.npz')
            target_scratch=os.path.join(scratch,'targets.npz')
            shutil.copy2(prediction_path,prediction_scratch)
            shutil.copy2(target_path,target_scratch)
            
            pred=np.load(prediction_scratch)
            targ=np.load(target_scratch)
        except BadZipfile:
            print("Npz file is corroupt, make new")
            shutil.rmtree(output_path)
        else:
            target_list=[targ["train"],targ["val"],targ["test"]]
            predctions=[pred["train"],pred["val"],pred["test"]]

            names=["train","validation","test"]

            if os.path.exists(os.path.join(output_path,"Error_fluct_test.parquet")):
                if overwrite==True:
                    error_fluc,err= plots.error(target_list,target_type,names,predctions,output_path) 
                    print('saved data',flush=True)
                    del error_fluc,err
                else:
                    print("error-data allready exist, and overwrite is false",flush=True)
            else:
                error_fluc,err= plots.error(target_list,target_type,names,predctions,output_path)
                print('saved data',flush=True)
                del error_fluc,err


            if os.path.exists(os.path.join(output_path,"test_PDF.png")):
                if overwrite==True or overwrite_pics==True or overwrite_pdf==True:
                    train=pd.read_parquet(os.path.join(output_path,'Error_fluct_train.parquet'))
                    val=pd.read_parquet(os.path.join(output_path,'Error_fluct_validation.parquet'))
                    test=pd.read_parquet(os.path.join(output_path,'Error_fluct_test.parquet'))
                    plots.pdf_plots([train,val,test],names,output_path,target_type)
                    plots.threeD_plot(val,output_path)
            else:
                train=pd.read_parquet(os.path.join(output_path,'Error_fluct_train.parquet'))
                val=pd.read_parquet(os.path.join(output_path,'Error_fluct_validation.parquet'))
                test=pd.read_parquet(os.path.join(output_path,'Error_fluct_test.parquet'))
                plots.pdf_plots([train,val,test],names,output_path,target_type)
                plots.threeD_plot(val,output_path)

            
            if os.path.exists(os.path.join(output_path,"target_prediction.pdf")):
                if overwrite==True or overwrite_pics==True:
                    if target[0]=="pr0.71_flux":
                        plots.heatmap_quarter(predctions,target_list,output_path,target)
                    elif target[0]=="tau_wall":
                        plots.heatmap_quarter(predctions,target_list,output_path,target)
                    else:
                        plots.heatmap_quarter_test(predctions[0],target_list[0],output_path,target)
                else:
                    print('heatmaps allready exist and overwrite is false',flush=True)        
            else:
                if target[0]=="pr0.71_flux":
                    plots.heatmap_quarter(predctions,target_list,output_path,target)
                elif target[0]=="tau_wall":
                    plots.heatmap_quarter(predctions,target_list,output_path,target)
                else:
                    plots.heatmap_quarter_test(predctions[0],target_list[0],output_path,target)

        plt.close('all')
        print("done with " +model,flush=True)


#%%

#output_path
#importlib.reload(plots)
#plots.heatmap_quarter_test(predctions[0],target_list[0],output_path,target)


