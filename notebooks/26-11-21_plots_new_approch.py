

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

#importlib.reload(plots)

overwrite=False
overwrite_pics=True


path_of_output="/home/au643300/DataHandling/models/output"
name_list, _ = utility.get_runs_wandb()

slurm_arrary_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))


#%%


model=name_list[slurm_arrary_id]
#model=name_list[0]

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


            if os.path.exists(os.path.join(output_path,"test_PDF.pdf")):
                if overwrite==True or overwrite_pics==True:
                    train=pd.read_parquet(os.path.join(output_path,'Error_fluct_train.parquet'))
                    val=pd.read_parquet(os.path.join(output_path,'Error_fluct_validation.parquet'))
                    test=pd.read_parquet(os.path.join(output_path,'Error_fluct_test.parquet'))
                    plots.pdf_plots([train,val,test],names,output_path,target_type)
            else:
                train=pd.read_parquet(os.path.join(output_path,'Error_fluct_train.parquet'))
                val=pd.read_parquet(os.path.join(output_path,'Error_fluct_validation.parquet'))
                test=pd.read_parquet(os.path.join(output_path,'Error_fluct_test.parquet'))
                plots.pdf_plots([train,val,test],names,output_path,target_type)

            
            if os.path.exists(os.path.join(output_path,"target_prediction.pdf")):
                if overwrite==True or overwrite_pics==True:
                    plots.heatmap_quarter(predctions,target_list,output_path,target)
                else:
                    print('heatmaps allready exist and overwrite is false',flush=True) 
            else:
                plots.heatmap_quarter(predctions,target_list,output_path,target)


        print("done with " +model,flush=True)



#%%
import seaborn as sns
import KDEpy



#Root sq. error of local heat flux
#'Root sq. error of local fluctuations'

# x_grid = np.linspace(1*10**-7, 1600, num=3**10)
# y_fluct = KDEpy.FFTKDE(bw='ISJ', kernel='gaussian').fit(val['Root sq. error of local fluctuations'].to_numpy(), weights=None).evaluate(x_grid)
# y_local = KDEpy.FFTKDE(bw='ISJ', kernel='gaussian').fit(val['Root sq. error of local heat flux'].to_numpy(), weights=None).evaluate(x_grid)

# #%%

# sns.set_theme()
# sns.set_context("paper")
# sns.set_style("ticks")

# f, ax = plt.subplots()
# #ax.set(xscale='log')


# sns.lineplot(x=x_grid, y=y_fluct, label='Root sq. error of local fluctuations',ax=ax)
# sns.lineplot(x=x_grid, y=y_local, label='Root sq. error of local heat flux',ax=ax)

# sns.despine()
# ax.set(xscale='log')
# ax.set_xlim(1*10**(-4),10**3)

# plt.fill_between(x_grid,y_fluct,alpha=0.8,color='grey')
# plt.fill_between(x_grid,y_local,alpha=0.4,color='grey')
# # plt.legend()

# # plt.show()
# # #plt.savefig("plot",bbox_inches="tight")

