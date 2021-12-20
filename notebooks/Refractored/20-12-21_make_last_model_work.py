

#%%

import os
import matplotlib
import importlib
from tensorflow import keras

import numpy as np
from glob import glob
from DataHandling import utility
from DataHandling import plots
from zipfile import BadZipfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from DataHandling.models import predict


name_list, _ = utility.get_runs_wandb()


#%%

model=name_list[1]
path_of_output="/home/au643300/DataHandling/models/output"
full_dir=os.path.join(path_of_output,model)
subdirs=os.listdir(full_dir)

dir=subdirs[-2]

#%%

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


pred=np.load("/home/au643300/DataHandling/models/output/twilight-aardvark-71/y_plus_15-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux/predictions.npz")
targ=np.load("/home/au643300/DataHandling/models/output/twilight-aardvark-71/y_plus_15-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux/targets.npz")


target_list=[targ["train"],targ["val"],targ["test"]]
predctions=[pred["train"],pred["val"],pred["test"]]

names=["train","validation","test"]



#train=pd.read_parquet(os.path.join(output_path,'Error_fluct_train.parquet'))
#val=pd.read_parquet(os.path.join(output_path,'Error_fluct_validation.parquet'))
#test=pd.read_parquet(os.path.join(output_path,'Error_fluct_test.parquet'))


#%%

#change the scale to plus units
Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu
Q_avg=0.665

cm =1/2.54

#max length in plus units
x_range=12/2
z_range=6/2

gridpoints_x=int(255/2)+1
gridponts_z=int(255/2)+1


x_plus_max=x_range*u_tau/nu
z_plus_max=z_range*u_tau/nu


x_plus_max=np.round(x_plus_max).astype(int)
z_plus_max=np.round(z_plus_max).astype(int)

axis_range_x=np.array([0,470,950,1420,1900,2370])
axis_range_z=np.array([0,295,590,890,1185])


placement_x=axis_range_x*nu/u_tau
placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)


placement_z=axis_range_z*nu/u_tau
placement_z=np.round((placement_z-0)/(z_range-0)*(gridponts_z-0)).astype(int)

Q_avg=0.665
a=target_list[0][5,:,:]/Q_avg
a=a[:128,:128]

b=predctions[0][5,:,:]/Q_avg
b=b[:128,:128]
#%%

fig2, ax=plt.subplots(1,figsize=([7*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)

pcm=ax.imshow(np.transpose(b),cmap='viridis',aspect=0.5,interpolation='bicubic')
ax.set_xlabel(r'$x^+$')
ax.set_ylabel(r'$z^+$')
#ax.set_title("test".capitalize(),weight="bold")
ax.set_xticks(placement_x,)
ax.set_xticklabels(axis_range_x,rotation=45)
ax.set_xticks(placement_x)
ax.set_xticklabels(axis_range_x,rotation=45)

ax.set_yticks(placement_z)
ax.set_yticklabels(axis_range_z)
cbar=fig2.colorbar(pcm,ax=ax,aspect=20,shrink=0.9,orientation="horizontal",pad=0.23)
cbar.formatter.set_powerlimits((0, 0))

if target[0]=='tau_wall':
    cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
elif target[0]=='pr0.71_flux':
    cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q}},\quad Pr=0.71$',rotation=0)
elif target[0]=='pr1_flux':
    cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q}},\quad Pr=1$',rotation=0)
elif target[0]=='pr0.2_flux':
    cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q}},\quad Pr=0.2$',rotation=0)
elif target[0]=='pr0.025_flux':
    cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q}},\quad Pr=0.025$',rotation=0)
else: 
    raise Exception('target name is not defined')

# ax.text(-0.35, 0.20, 'Target',
#        verticalalignment='bottom', horizontalalignment='right',
#        transform=ax.transAxes,rotation=90,weight="bold")

ax.text(-0.35, 0.1, 'Prediction',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,rotation=90,weight="bold")



#fig2.savefig(os.path.join(output_path,'test_target.pdf'),bbox_inches='tight',format='pdf')
fig2.savefig(os.path.join(output_path,'test_prediction.pdf'),bbox_inches='tight',format='pdf')



#%%

d=plt.imshow(np.transpose(b),cmap='viridis',aspect=0.5,interpolation='bicubic')
plt.colorbar(d)



#%%


name="twilight-aardvark-71"
overwrite=False
vars=["u_vel","v_vel","w_vel","pr0.025"]
target=["pr0.025_flux"]
normalize=False
model=keras.models.load_model("/home/au643300/DataHandling/models/trained/twilight-aardvark-71")
y_plus=15
pr_number="pr0.025"
for layer in model.layers:
    if layer.name[0:2]=="pr":
        layer._name=pr_number
#%%

predict.predict(name,overwrite,model,y_plus,vars,target,normalize)

