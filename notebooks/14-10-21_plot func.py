



#%%
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from DataHandling.features import slices
import shutil



#%%

def heatmaps(model_name,var,target,y_plus,normalize):
    """Makes heatmaps of train,validation and test and compares them with the target. Also outputs the difference between prediciton and target

    Args:
        model_name (Wandb Name): Name given to run by wandb
        var (list): list of variables without target
        target (list): list of target
        y_plus (int): y_plus value used for planes
        normalize (bool): if the data is normalized or not

    Returns:
        None: Saves the figures in the reports folder
    """
    from DataHandling.utility import get_data
    from DataHandling.features import slices
    from tensorflow import keras
    import numpy as np
    import shutil
    import os
    import matplotlib.pyplot as plt

    model_path=os.path.join("/home/au643300/DataHandling/models/trained/",model_name)
    output_path='/home/au643300/DataHandling/reports/figures'
    data_path=slices.slice_loc(y_plus,var,target,normalize)+"/"
    data_folder=os.path.basename(os.path.dirname(data_path))

    output_path=os.path.join(output_path,model_name+'_'+data_folder)

    data=slices.load_from_scratch(y_plus,var,target,normalized=normalize)


    model=keras.models.load_model(model_path)



    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        print('deleting old version')
        shutil.rmtree(output_path)           
        os.makedirs(output_path)

    feature_list,target_list,names=get_data(data)


    predctions=[]
    for features in feature_list:
        predctions.append(model.predict(features))

    predctions=[np.squeeze(x,axis=3) for x in predctions]

    #Find highest and lowest value to scale plot to


    max_tot=0
    min_tot=1000
    for i in range(3):
        max_inter=np.max([np.max(target_list[i][1,:,:]),np.max(predctions[i][1,:,:])])
        min_inter=np.min([np.min(target_list[i][1,:,:]),np.min(predctions[i][1,:,:])])
        
        
        if max_inter>max_tot:
            max_tot=max_inter
        if min_inter<min_tot:
            min_tot=min_inter


    fig, axs=plt.subplots(2,3,sharex=True,sharey=True,constrained_layout=False)
    #To display the correct axis on the plot
    axis_range=np.linspace(0,255,4)
    x_axis_range=(axis_range-0)/(255-0)*(12-0)+0
    x_axis_range=np.round(x_axis_range).astype(int)
    z_axis_range=(axis_range-0)/(255-0)*(6-0)+0
    z_axis_range=np.round(z_axis_range).astype(int)
    z_axis_range=np.flip(z_axis_range)
    for i in range(3):     
        #Target
        pcm=axs[0,i].imshow(target_list[i][1,:,:],cmap='inferno',vmin=min_tot,vmax=max_tot)
        axs[0,i].set_title(names[i].capitalize(),weight="bold")
        axs[0,0].set_ylabel(r'$z/h$')
        
        #prediction
        axs[1,i].imshow(predctions[i][1,:,:],cmap='inferno',vmin=min_tot,vmax=max_tot)
        axs[1,i].set_xlabel(r'$x/h$')
        axs[1,0].set_ylabel(r'$z/h$')

        axs[1,i].set_xticks(axis_range)
        axs[1,i].set_xticklabels(x_axis_range)
        axs[0,0].set_yticks(axis_range)
        axs[0,0].set_yticklabels(z_axis_range)
        axs[1,0].set_yticks(axis_range)
        axs[1,0].set_yticklabels(z_axis_range)

        
    #Setting labels and stuff
    axs[0,0].text(-0.5, 0.30, 'Target',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[0,0].transAxes,rotation=90,weight="bold")

    axs[1,0].text(-0.5, 0.20, 'Prediction',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1,0].transAxes,rotation=90,weight="bold")

    fig.subplots_adjust(wspace=-0.51,hspace=0.18)
    cbar=fig.colorbar(pcm,ax=axs[:,:],aspect=20,shrink=0.5,location="bottom")
    cbar.formatter.set_powerlimits((0, 0))

    #TODO skal ændres sådan den ved om det er tau_wall eller andet som er på heatmap
    cbar.ax.set_xlabel(r'$\tau_{wall } $',rotation=0)
    fig.savefig(os.path.join(output_path,'target_prediction.pdf'),bbox_inches='tight',dpi=100,format='pdf')


    max_diff=np.max([np.max(target_list[0][1,:,:]-predctions[0][1,:,:]),np.max(target_list[1][1,:,:]-predctions[1][1,:,:]),np.max(target_list[2][1,:,:]-predctions[2][1,:,:])])
    min_diff=np.min([np.min(target_list[0][1,:,:]-predctions[0][1,:,:]),np.min(target_list[1][1,:,:]-predctions[1][1,:,:]),np.min(target_list[2][1,:,:]-predctions[2][1,:,:])])

    fig2, axs=plt.subplots(1,3,sharex=True,sharey=True,constrained_layout=True)
    for i in range(3):
        pcm=axs[i].imshow(target_list[i][1,:,:]-predctions[i][1,:,:],cmap="Spectral",vmin=min_diff,vmax=max_diff)
        axs[i].set_xlabel(r'$x/h$')
        axs[0].set_ylabel(r'$z/h$')
        axs[i].set_title(names[i].capitalize(),weight="bold")
    cbar=fig.colorbar(pcm,ax=axs[:],aspect=15,shrink=0.5,location="bottom")
    cbar.ax.set_xlabel(r'Difference $\tau_{wall } $',rotation=0)
    cbar.formatter.set_powerlimits((0, 0))

    fig2.savefig(os.path.join(output_path,'difference.pdf'),bbox_inches='tight',dpi=200,format='pdf')

    keras.utils.plot_model(model,to_file=os.path.join(output_path,"network.png"),show_shapes=True,dpi=100)


    return None



var=['u_vel']
target=['tau_wall']
normalize=False
y_plus=15
model_name="lively-monkey-15"

heatmaps(model_name,var,target,y_plus,normalize)



#%%



from DataHandling.utility import get_data
from DataHandling.features import slices
from tensorflow import keras
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt

model_path=os.path.join("/home/au643300/DataHandling/models/trained/",model_name)
output_path='/home/au643300/DataHandling/reports/figures'
data_path=slices.slice_loc(y_plus,var,target,normalize)+"/"
data_folder=os.path.basename(os.path.dirname(data_path))

output_path=os.path.join(output_path,model_name+'_'+data_folder)

data=slices.load_from_scratch(y_plus,var,target,normalized=normalize)


model=keras.models.load_model(model_path)



if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    print('deleting old version')
    shutil.rmtree(output_path)           
    os.makedirs(output_path)

feature_list,target_list,names=get_data(data)


predctions=[]
for features in feature_list:
    predctions.append(model.predict(features))

predctions=[np.squeeze(x,axis=3) for x in predctions]

#Find highest and lowest value to scale plot to


max_tot=0
min_tot=1000
for i in range(3):
    max_inter=np.max([np.max(target_list[i][1,:,:]),np.max(predctions[i][1,:,:])])
    min_inter=np.min([np.min(target_list[i][1,:,:]),np.min(predctions[i][1,:,:])])
    
    
    if max_inter>max_tot:
        max_tot=max_inter
    if min_inter<min_tot:
        min_tot=min_inter


fig, axs=plt.subplots(2,3,sharex=True,sharey=True,constrained_layout=False)
#To display the correct axis on the plot
axis_range=np.linspace(0,255,4)
x_axis_range=(axis_range-0)/(255-0)*(12-0)+0
x_axis_range=np.round(x_axis_range).astype(int)
z_axis_range=(axis_range-0)/(255-0)*(6-0)+0
z_axis_range=np.round(z_axis_range).astype(int)
z_axis_range=np.flip(z_axis_range)
for i in range(3):     
    #Target
    pcm=axs[0,i].imshow(target_list[i][1,:,:],cmap='inferno',vmin=min_tot,vmax=max_tot)
    axs[0,i].set_title(names[i].capitalize(),weight="bold")
    axs[0,0].set_ylabel(r'$z/h$')
    
    #prediction
    axs[1,i].imshow(predctions[i][1,:,:],cmap='inferno',vmin=min_tot,vmax=max_tot)
    axs[1,i].set_xlabel(r'$x/h$')
    axs[1,0].set_ylabel(r'$z/h$')

    axs[1,i].set_xticks(axis_range)
    axs[1,i].set_xticklabels(x_axis_range)
    axs[0,0].set_yticks(axis_range)
    axs[0,0].set_yticklabels(z_axis_range)
    axs[1,0].set_yticks(axis_range)
    axs[1,0].set_yticklabels(z_axis_range)

    
#Setting labels and stuff
axs[0,0].text(-0.5, 0.30, 'Target',
        verticalalignment='bottom', horizontalalignment='right',
        transform=axs[0,0].transAxes,rotation=90,weight="bold")

axs[1,0].text(-0.5, 0.20, 'Prediction',
        verticalalignment='bottom', horizontalalignment='right',
        transform=axs[1,0].transAxes,rotation=90,weight="bold")

fig.subplots_adjust(wspace=-0.51,hspace=0.18)
cbar=fig.colorbar(pcm,ax=axs[:,:],aspect=20,shrink=0.5,location="bottom")
cbar.formatter.set_powerlimits((0, 0))

#TODO skal ændres sådan den ved om det er tau_wall eller andet som er på heatmap
cbar.ax.set_xlabel(r'$\tau_{wall } $',rotation=0)
fig.savefig(os.path.join(output_path,'target_prediction.pdf'),bbox_inches='tight',dpi=100,format='pdf')


max_diff=np.max([np.max(target_list[0][1,:,:]-predctions[0][1,:,:]),np.max(target_list[1][1,:,:]-predctions[1][1,:,:]),np.max(target_list[2][1,:,:]-predctions[2][1,:,:])])
min_diff=np.min([np.min(target_list[0][1,:,:]-predctions[0][1,:,:]),np.min(target_list[1][1,:,:]-predctions[1][1,:,:]),np.min(target_list[2][1,:,:]-predctions[2][1,:,:])])

fig2, axs=plt.subplots(1,3,sharex=True,sharey=True,constrained_layout=True)
for i in range(3):
    pcm=axs[i].imshow(target_list[i][1,:,:]-predctions[i][1,:,:],cmap="Spectral",vmin=min_diff,vmax=max_diff)
    axs[i].set_xlabel(r'$x/h$')
    axs[0].set_ylabel(r'$z/h$')
    axs[i].set_title(names[i].capitalize(),weight="bold")
cbar=fig.colorbar(pcm,ax=axs[:],aspect=15,shrink=0.5,location="bottom")
cbar.ax.set_xlabel(r'Difference $\tau_{wall } $',rotation=0)
cbar.formatter.set_powerlimits((0, 0))

fig2.savefig(os.path.join(output_path,'difference.pdf'),bbox_inches='tight',dpi=200,format='pdf')

keras.utils.plot_model(model,to_file=os.path.join(output_path,"network.png"),show_shapes=True,dpi=100)


# %%
