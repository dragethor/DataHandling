
#%%




""" Det her er et ufærdigt script som jeg har brugt til at tjekke error rumligt. Alt det som ikke er i en func er taget direkte fra plots_new_approch """


import os
import matplotlib
import importlib
#matplotlib.use('Agg')


import numpy as np
from glob import glob
from DataHandling import utility
from DataHandling import plots
from zipfile import BadZipfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


overwrite=False
overwrite_pics=True


path_of_output="/home/au643300/DataHandling/models/output"
name_list, _ = utility.get_runs_wandb()

#slurm_arrary_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))

#model=name_list[slurm_arrary_id]
importlib.reload(plots)
model=name_list[0]

full_dir=os.path.join(path_of_output,model)
subdirs=os.listdir(full_dir)

print('This is model ' + model,flush=True)

dir=subdirs[-1]


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

#%%
train=pd.read_parquet(os.path.join(output_path,'Error_fluct_train.parquet'))
val=pd.read_parquet(os.path.join(output_path,'Error_fluct_validation.parquet'))
test=pd.read_parquet(os.path.join(output_path,'Error_fluct_test.parquet'))

#%%
train_numpy=train.to_numpy()
reshape_t=train_numpy.reshape((814,256,256))
avg=np.median(reshape_t,0)

#%%

from scipy.ndimage.filters import gaussian_filter

data = gaussian_filter(avg, 1)
plt.contour(np.transpose(data))



#%%

def threeD_plot(error_val,names,output_path):
    #change the scale to plus units
    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu


    train_numpy=error_val.to_numpy()
    num_snapshots=int(train_numpy/256/256)
    reshape_t=train_numpy.reshape((num_snapshots,256,256))
    avg=np.mean(reshape_t,0)

    # Create meshgrid
    xx, yy = np.mgrid[0:256:256j, 0:256:256j]


    x_range=12
    z_range=6

    gridpoints_x=int(255)+1
    gridponts_z=int(255)+1


    x_plus_max=x_range*u_tau/nu
    z_plus_max=z_range*u_tau/nu


    x_plus_max=np.round(x_plus_max).astype(int)
    z_plus_max=np.round(z_plus_max).astype(int)

    axis_range_x=np.array([0,950,1900,2850,3980,4740])
    axis_range_z=np.array([0,470,950,1420,1900,2370])


    placement_x=axis_range_x*nu/u_tau
    placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)


    placement_z=axis_range_z*nu/u_tau
    placement_z=np.round((placement_z-0)/(z_range-0)*(gridponts_z-0)).astype(int)




    cm =1/2.54
    fig = plt.figure(figsize=(15*cm,10*cm),dpi=200)
    ax = plt.axes(projection='3d')
    matplotlib.colors.Normalize
    surf = ax.plot_surface(xx, yy, np.transpose(avg), cmap='viridis', edgecolor='none')
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$z^+$')
    ax.set_zlabel(r'Error $\%$')
    ax.set_box_aspect((2,1,1))

    ax.set_xticks(placement_x,)
    ax.set_xticklabels(axis_range_x,)
    ax.set_xticks(placement_x)
    ax.set_xticklabels(axis_range_x)

    ax.set_yticks(placement_z)
    ax.set_yticklabels(axis_range_z)
    fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
    ax.view_init(30, 140)
    
    fig.savefig(os.path.join(output_path,'validation_3D.pdf'),bbox_inches='tight')