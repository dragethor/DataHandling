#%%


def heatmap_quarter_test(predction,target_var,output_path,target):
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
    name='test'
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
        target_var=target_var[1,:,:]/u_tau**2
        predction=predction[1,:,:]/u_tau**2        

        #cut the data to 1/4
        target_var=target_var[:128,:128]
        predction=predction[:128,:128]
            
    elif target[0][-5:]=='_flux':
        fric_temp=Q_avg/u_tau
        target_var=target_var[1,:,:]/Q_avg
        predction=predction[1,:,:]/Q_avg  

        #cut the data to 1/4
        target_var=target_var[:128,:128]
        predction=predction[:128,:128]

        #Need to find the average surface heat flux Q_w
        #Friction temp = Q_w/(u_tau)
        #q^+= q/(Friction temp)



    #Find highest and lowest value to scale plot to
    max_tot=np.max([np.max(target_var),np.max(predction)])
    min_tot=np.min([np.min(target_var),np.min(predction)])



    



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

    
    fig, axs=plt.subplots(2,figsize=([7*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=300)

    #Target
    pcm=axs[0].imshow(np.transpose(target_var),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
    axs[0].set_title(name.capitalize(),weight="bold")
    axs[0].set_ylabel(r'$z^+$')
    
    #prediction
    axs[1].imshow(np.transpose(predction),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
    axs[1].set_xlabel(r'$x^+$')
    axs[1].set_ylabel(r'$z^+$')

    axs[1].set_xticks(placement_x)
    axs[1].set_xticklabels(axis_range_x,rotation=45)
    axs[0].set_yticks(placement_z)
    axs[0].set_yticklabels(axis_range_z)
    axs[1].set_yticks(placement_z)
    axs[1].set_yticklabels(axis_range_z)

        
    #Setting labels and stuff
    axs[0].text(-0.42, 0.20, 'Target',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[0].transAxes,rotation=90,weight="bold")

    axs[1].text(-0.42, 0.00, 'Prediction',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1].transAxes,rotation=90,weight="bold")

    fig.subplots_adjust(wspace=-0.31,hspace=0.25)
    cbar=fig.colorbar(pcm,ax=axs[:],aspect=20,shrink=0.7,location="bottom",pad=0.22)
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

    fig.savefig(os.path.join(output_path,'target_prediction.pdf'),bbox_inches='tight',format='pdf')


    max_diff=np.max(target_var-predction)
    min_diff=np.min(target_var-predction)



    fig2, ax=plt.subplots(1,figsize=([7*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=300)

    pcm=ax.imshow(np.transpose(target_var-predction),cmap="Spectral",vmin=min_diff,vmax=max_diff,aspect=0.5)
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$z^+$')
    ax.set_title("difference".capitalize(),weight="bold")
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

    fig2.savefig(os.path.join(output_path,'difference.pdf'),bbox_inches='tight',format='pdf')




#%%
import numpy as np



pre_path="/home/au643300/DataHandling/models/output/copper-pyramid-66/y_plus_15-VARS-u_vel_v_vel_w_vel-TARGETS-pr0.025_flux/predictions.npz"
target_path="/home/au643300/DataHandling/models/output/copper-pyramid-66/y_plus_15-VARS-u_vel_v_vel_w_vel-TARGETS-pr0.025_flux/targets.npz"


pred=np.load(pre_path)
targ=np.load(target_path)


pred1=pred["train"]
targ1=targ["train"]

target=['pr0.025_flux']
#%%
#,pad=0.20
a=heatmap_quarter_test_only(pred1,targ1,"output_path",target)





