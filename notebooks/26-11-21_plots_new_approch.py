

#%%
import os
import matplotlib

matplotlib.use('PDF')


import numpy as np
from glob import glob
from DataHandling import utility
from DataHandling import plots
from zipfile import BadZipfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt

overwrite=False
path_of_output="/home/au643300/DataHandling/models/output"
model_names=[ f.name for f in os.scandir(path_of_output) if f.is_dir() ]


slurm_arrary_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))


#%%




model=model_names[slurm_arrary_id]

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

            if os.path.exists(os.path.join(output_path,"target_prediction.pdf")):
                if overwrite==True:
                    plots.heatmaps(target_list,names,predctions,output_path,model_path,target)
                else:
                    print('heatmaps allready exist and overwrite is false',flush=True) 
            else:
                plots.heatmaps(target_list,names,predctions,output_path,model_path,target)



            if os.path.exists(os.path.join(output_path,"Error_fluct_test.csv")):
                if overwrite==True:
                    error_fluc,err= plots.error(target_list,target_type,names,predctions,output_path) 
                    print('saved data',flush=True)
                    del error_fluc,err
                else:
                    print("Error data allready exist, and overwrite is false",flush=True)
            else:
                error_fluc,err= plots.error(target_list,target_type,names,predctions,output_path)
                print('saved data',flush=True)
                del error_fluc,err


            if os.path.exists(os.path.join(output_path,"test_PDF.pdf")):
                if overwrite==True:
                    plots.pdf_plots(error_fluc,names,output_path)
            else:
                train1=pd.read_csv(os.path.join(output_path,'Error_fluct_train.csv'))
                val1=pd.read_csv(os.path.join(output_path,'Error_fluct_validation.csv'))
                test1=pd.read_csv(os.path.join(output_path,'Error_fluct_test.csv'))
                
                train1.to_parquet(os.path.join(output_path,'Error_fluct_'+"train"+'.parquet'))
                val1.to_parquet(os.path.join(output_path,'Error_fluct_'+"validation"+'.parquet'))
                test1.to_parquet(os.path.join(output_path,'Error_fluct_'+"test"+'.parquet'))

                os.remove(os.path.join(output_path,'Error_fluct_train.csv'))
                os.remove(os.path.join(output_path,'Error_fluct_validation.csv'))
                os.remove(os.path.join(output_path,'Error_fluct_test.csv'))
                #train_csv=pd.read_parquet(os.path.join(output_path,'Error_fluct_train.csv'))
                #val_csv=pd.read_parquet(os.path.join(output_path,'Error_fluct_validation.csv'))
                #test_csv=pd.read_parquet(os.path.join(output_path,'Error_fluct_test.csv'))
                #plots.pdf_plots([train_csv,val_csv,test_csv],names,output_path)

        print("done with " +model,flush=True)



#%%
import seaborn as sns

import KDEpy



#Root sq. error of local heat flux
#'Root sq. error of local fluctuations'

# x_grid = np.linspace(0, 1000, num=2**10)
# x_fluct, y_fluct = KDEpy.FFTKDE(bw='ISJ', kernel='gaussian').fit(val_csv['Root sq. error of local fluctuations'].to_numpy(), weights=None).evaluate(x_grid)
# x_local, y_local = KDEpy.FFTKDE(bw='ISJ', kernel='gaussian').fit(val_csv['Root sq. error of local heat flux'].to_numpy(), weights=None).evaluate(x_grid)

# #%%



# plt.cla()
# #plt.plot(x_fluct, y_fluct, label='Root sq. error of local fluctuations')
# plt.plot(x_local, y_local, label="Root sq. error of local heat flux")
# plt.xscale('log')


# #plt.xlim(1*10**(-2),1*10**(3))
# sns.despine()
# #plt.fill_between(x_fluct,y_fluct,alpha=0.4,color='grey')
# plt.fill_between(x_local,y_local,alpha=0.8,color='grey')
# plt.legend()

# plt.show()
# #plt.savefig("plot",bbox_inches="tight")

