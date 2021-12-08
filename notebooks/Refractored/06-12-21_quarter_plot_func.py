
#%%

# import numpy as np


# #%%

# targ=np.load('/home/au643300/DataHandling/models/output/icy-fire-52/y_plus_15-VARS-u_vel-TARGETS-pr0.71_flux/targets.npz',allow_pickle=True)
# pred=np.load('/home/au643300/DataHandling/models/output/icy-fire-52/y_plus_15-VARS-u_vel-TARGETS-pr0.71_flux/predictions.npz',allow_pickle=True)

# target_list=[targ["train"],targ["val"],targ["test"]]
# predctions=[pred["train"],pred["val"],pred["test"]]
# names=['train','validation','test']

# output_path='/home/au643300/DataHandling/models/output/icy-fire-52/y_plus_15-VARS-u_vel-TARGETS-pr0.71_flux/'

# target=["pr0.71_flux"]




#%%
def heatmap_quarter(predctions,target_list,output_path,target):
    from DataHandling import utility
    from DataHandling.features import slices
    from tensorflow import keras
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("paper")
    names=['train',"validation",'test']
    cm = 1/2.54  # centimeters in inches




    if not os.path.exists(output_path):
        os.makedirs(output_path)


    #change the scale to plus units
    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    Q_avg=0.665





    if target[0]=='tau_wall':
        for i in range(len(target_list)):
            target_list[i]=target_list[i][1,:,:]/u_tau**2
            predctions[i]=predctions[i][1,:,:]/u_tau**2        

            #cut the data to 1/4
            target_list[i]=target_list[i][:128,:128]
            predctions[i]=predctions[i][:128,:128]
            
    elif target[0][-5:]=='_flux':
        fric_temp=Q_avg/u_tau
        for i in range(len(target_list)):
            target_list[i]=target_list[i][1,:,:]/Q_avg
            predctions[i]=predctions[i][1,:,:]/Q_avg  

            #cut the data to 1/4
            target_list[i]=target_list[i][:128,:128]
            predctions[i]=predctions[i][:128,:128]

        #Need to find the average surface heat flux Q_w
        #Friction temp = Q_w/(u_tau)
        #q^+= q/(Friction temp)



    #Find highest and lowest value to scale plot to
    max_tot=0
    min_tot=1000
    for i in range(3):
        max_inter=np.max([np.max(target_list[i]),np.max(predctions[i])])
        min_inter=np.min([np.min(target_list[i]),np.min(predctions[i])])
        
        
        if max_inter>max_tot:
            max_tot=max_inter
        if min_inter<min_tot:
            min_tot=min_inter


    fig, axs=plt.subplots(2,3,figsize=([21*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=150)



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

    for i in range(3):  

        #Target
        pcm=axs[0,i].imshow(np.transpose(target_list[i]),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
        axs[0,i].set_title(names[i].capitalize(),weight="bold")
        axs[0,0].set_ylabel(r'$z^+$')
        
        #prediction
        axs[1,i].imshow(np.transpose(predctions[i]),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
        axs[1,i].set_xlabel(r'$x^+$')
        axs[1,0].set_ylabel(r'$z^+$')

        axs[1,0].set_xticks(placement_x)
        axs[1,0].set_xticklabels(axis_range_x,rotation=45)
        axs[1,1].set_xticks(placement_x)
        axs[1,1].set_xticklabels(axis_range_x,rotation=45)
        axs[1,2].set_xticks(placement_x)
        axs[1,2].set_xticklabels(axis_range_x,rotation=45)
        axs[0,0].set_yticks(placement_z)
        axs[0,0].set_yticklabels(axis_range_z)
        axs[1,0].set_yticks(placement_z)
        axs[1,0].set_yticklabels(axis_range_z)

        
    #Setting labels and stuff
    axs[0,0].text(-0.45, 0.20, 'Target',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[0,0].transAxes,rotation=90,weight="bold")

    axs[1,0].text(-0.45, 0.00, 'Prediction',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1,0].transAxes,rotation=90,weight="bold")

    fig.subplots_adjust(wspace=-0.31,hspace=0.25)
    cbar=fig.colorbar(pcm,ax=axs[:,:],aspect=30,shrink=0.55,location="bottom",pad=0.24)
    cbar.formatter.set_powerlimits((0, 0))


    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$q_w^+,\quad Pr=0.71$',rotation=0)
    else: 
        raise Exception('target name is not defined')

    fig.savefig(os.path.join(output_path,'target_prediction.pdf'),bbox_inches='tight',format='pdf')


    max_diff=np.max([np.max(target_list[0]-predctions[0]),np.max(target_list[1]-predctions[1]),np.max(target_list[2]-predctions[2])])
    min_diff=np.min([np.min(target_list[0]-predctions[0]),np.min(target_list[1]-predctions[1]),np.min(target_list[2]-predctions[2])])

    fig2, axs=plt.subplots(1,3,figsize=([21*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=150)
    for i in range(3):
        pcm=axs[i].imshow(target_list[i]-predctions[i],cmap="Spectral",vmin=min_diff,vmax=max_diff,aspect=0.5)
        axs[i].set_xlabel(r'$x^+$')
        axs[0].set_ylabel(r'$z^+$')
        axs[i].set_title(names[i].capitalize(),weight="bold")
        axs[0].set_xticks(placement_x,)
        axs[0].set_xticklabels(axis_range_x,rotation=45)
        axs[1].set_xticks(placement_x)
        axs[1].set_xticklabels(axis_range_x,rotation=45)
        axs[2].set_xticks(placement_x)
        axs[2].set_xticklabels(axis_range_x,rotation=45)

    axs[0].set_yticks(placement_z)
    axs[0].set_yticklabels(axis_range_z)
    fig2.subplots_adjust(wspace=0.13,hspace=0.05)
    cbar=fig.colorbar(pcm,ax=axs[:],aspect=30,shrink=0.55,location="bottom",pad=0.18)
    cbar.formatter.set_powerlimits((0, 0))

    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$q_w^+,\quad Pr=0.71$',rotation=0)
    else: 
        raise Exception('target name is not defined')



    fig2.savefig(os.path.join(output_path,'difference.pdf'),bbox_inches='tight',format='pdf')
        
