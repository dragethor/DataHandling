


def pdf_plots(error_fluc,names,output_path):
    """Makes both boxplot and pdf plot for the errors.

    Args:
        error_fluc (list): list of the train,validation,test errors in local form
        names (list): list of the names of the data. Normally train,validaiton,test
        output_path (Path): Path of where to save the figures
    """
    from DataHandling import utility
    from DataHandling.features import slices
    from tensorflow import keras
    import numpy as np
    import shutil
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    for i in range(3):
        cm =1/2.54
        fig, ax = plt.subplots(1, 1,figsize=(20*cm,20*cm),dpi=200)
        sns.boxplot(data=error_fluc[i],showfliers = False,orient='h',ax=ax)
        fig.savefig(os.path.join(output_path,names[i]+'_boxplot.pdf'),bbox_inches='tight',format='pdf')

        plt.figure(figsize=(20*cm,10*cm),dpi=200)
        sns.histplot(data=error_fluc[i],stat='density',kde=True,log_scale=True)
        plt.xlim(1*10**(-2),1*10**(3))
        plt.savefig(os.path.join(output_path,names[i]+'_PDF.pdf'),bbox_inches='tight',format='pdf')




def error(target_list,target_type,names,predctions,output_path):
    import os
    import numpy as np
    import pandas as pd
    

    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

 
    
    if target_type=="stress":
        error=pd.DataFrame(columns=['Global Mean Error','Root mean sq. error of local shear stress','Global fluctuations error','Root mean sq. error of local fluctuations'])
    elif target_type=="flux":
        error=pd.DataFrame(columns=['Global Mean Error','Root mean sq. error of local heat flux','Global fluctuations error','Root mean sq. error of local fluctuations'])
    
    error_fluc_list=[]
    
    for i in range(3):
        error_fluct=pd.DataFrame()
        fluc_predict=predctions[i][:,:,:]-np.mean(predctions[i][:,:,:])
        fluc_target=target_list[i][:,:,:]-np.mean(target_list[i][:,:,:])
        

        #Global average errors
        global_mean_err=(np.mean(predctions[i][:,:,:])-np.mean(target_list[i][:,:,:]))/(np.mean(target_list[i][:,:,:]))*100
        MSE_local_shear_stress=np.sqrt((np.mean((predctions[i][:,:,:]-target_list[i][:,:,:])**2))/np.mean(target_list[i][:,:,:])**2)*100
        global_fluct_error=(np.std(fluc_predict)-np.std(fluc_target))/(np.std(fluc_target))*100
        MSE_local_fluc=np.sqrt((np.mean(fluc_predict-fluc_target)**2)/np.std(fluc_target)**2)*100

        

        #MAE_local=np.mean(np.abs(predctions[i][:,:,:]-target_list[i][:,:,:]))/np.mean(np.abs(target_list[i][:,:,:]))*100
        #MAE_local_no_mean=(np.abs(predctions[i][:,:,:]-target_list[i][:,:,:]))/np.mean(np.abs(target_list[i][:,:,:]))*100
        #MAE_fluct_no_mean=(np.abs(fluc_predict-fluc_target))/np.mean(np.abs(fluc_target))*100
        

        

        #Local erros for PDF's and boxplots etc.
        MSE_local_no_mean=np.sqrt(((predctions[i][:,:,:]-target_list[i][:,:,:])**2)/np.mean(target_list[i][:,:,:])**2)*100
        MSE_local_fluc_PDF=np.sqrt(((fluc_predict-fluc_target)**2)/(np.std(fluc_target))**2)*100
        
        

        if target_type=="stress":
            error_fluct['Root sq. error of local shear stress']=MSE_local_no_mean.flatten()
            error=error.append({'Global Mean Error':global_mean_err,'Root mean sq. error of local shear stress':MSE_local_shear_stress,'Global fluctuations error':global_fluct_error,'Root mean sq. error of local fluctuations':MSE_local_fluc},ignore_index=True)
        elif target_type=="flux":
            error_fluct['Root sq. error of local heat flux']=MSE_local_no_mean.flatten()
            error=error.append({'Global Mean Error':global_mean_err,'Root mean sq. error of local shear stress':MSE_local_shear_stress,'Global fluctuations error':global_fluct_error,'Root mean sq. error of local fluctuations':MSE_local_fluc},ignore_index=True)
        
        error_fluct['Root sq. error of local fluctuations']=MSE_local_fluc_PDF.flatten()
        #error_fluct['MAE local']=MAE_local_no_mean.flatten()
        #error_fluct['MAE fluct']=MAE_fluct_no_mean.flatten()

        

        error_fluct.to_csv(os.path.join(output_path,'Error_fluct_'+names[i]+'.csv'))
        error_fluc_list.append(error_fluct)
        

    
    error.index=names

    error.to_csv(os.path.join(output_path,'Mean_error.csv'))

    return error_fluc_list, error


def heatmaps(target_list,names,predctions,output_path,model_path,target):
    """makes heatmaps of the Train validation and test data for target and prediction. Also plots the difference. Save to the output folder

    Args:
        target_list (list): list of arrays of the target
        names (list): list of names for the target_list
        predctions (list): list of array of the prediction
        output_path (Path): Path to the output folder
        model_path (Path): Path to the saved model

    Raises:
        Exception: if the target has no defined plot name
        Exception: Same as above

    Returns:
        None: 
    """
    from DataHandling import utility
    from DataHandling.features import slices
    from tensorflow import keras
    import numpy as np
    import shutil
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = 1/2.54  # centimeters in inches


    model=keras.models.load_model(model_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)


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


    fig, axs=plt.subplots(2,3,figsize=([21*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=150)

    #To display the correct axis on the plot
    axis_range_z=np.linspace(0,255,4)
    axis_range_x=np.linspace(0,255,7)
    x_axis_range=(axis_range_x-0)/(255-0)*(12-0)+0
    x_axis_range=np.round(x_axis_range).astype(int)
    z_axis_range=(axis_range_z-0)/(255-0)*(6-0)+0
    z_axis_range=np.round(z_axis_range).astype(int)
    z_axis_range=np.flip(z_axis_range)
    for i in range(3):  

        #Target
        pcm=axs[0,i].imshow(np.transpose(target_list[i][1,:,:]),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
        axs[0,i].set_title(names[i].capitalize(),weight="bold")
        axs[0,0].set_ylabel(r'$z/h$')
        
        #prediction
        axs[1,i].imshow(np.transpose(predctions[i][1,:,:]),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
        axs[1,i].set_xlabel(r'$x/h$')
        axs[1,0].set_ylabel(r'$z/h$')

        axs[1,i].set_xticks(axis_range_x)
        axs[1,i].set_xticklabels(x_axis_range)
        axs[0,0].set_yticks(axis_range_z)
        axs[0,0].set_yticklabels(z_axis_range)
        axs[1,0].set_yticks(axis_range_z)
        axs[1,0].set_yticklabels(z_axis_range)

        
    #Setting labels and stuff
    axs[0,0].text(-0.23, 0.30, 'Target',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[0,0].transAxes,rotation=90,weight="bold")

    axs[1,0].text(-0.23, 0.20, 'Prediction',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1,0].transAxes,rotation=90,weight="bold")

    fig.subplots_adjust(wspace=0.09,hspace=0.15)
    cbar=fig.colorbar(pcm,ax=axs[:,:],aspect=20,shrink=0.5,location="bottom")
    cbar.formatter.set_powerlimits((0, 0))


    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{wall } $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$\Pr_{wall} 0.71 $',rotation=0)
    else: 
        raise Exception('target name is not defined')

    fig.savefig(os.path.join(output_path,'target_prediction.pdf'),bbox_inches='tight',format='pdf')


    max_diff=np.max([np.max(target_list[0][1,:,:]-predctions[0][1,:,:]),np.max(target_list[1][1,:,:]-predctions[1][1,:,:]),np.max(target_list[2][1,:,:]-predctions[2][1,:,:])])
    min_diff=np.min([np.min(target_list[0][1,:,:]-predctions[0][1,:,:]),np.min(target_list[1][1,:,:]-predctions[1][1,:,:]),np.min(target_list[2][1,:,:]-predctions[2][1,:,:])])

    fig2, axs=plt.subplots(1,3,figsize=([21*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=150)
    for i in range(3):
        pcm=axs[i].imshow(target_list[i][1,:,:]-predctions[i][1,:,:],cmap="Spectral",vmin=min_diff,vmax=max_diff,aspect=0.5)
        axs[i].set_xlabel(r'$x/h$')
        axs[0].set_ylabel(r'$z/h$')
        axs[i].set_title(names[i].capitalize(),weight="bold")
        axs[i].set_xticks(axis_range_x)
        axs[i].set_xticklabels(x_axis_range)

    axs[0].set_yticks(axis_range_z)
    axs[0].set_yticklabels(z_axis_range)
    fig2.subplots_adjust(wspace=0.09,hspace=0.05)
    cbar=fig.colorbar(pcm,ax=axs[:],aspect=20,shrink=0.5,location="bottom")
    cbar.formatter.set_powerlimits((0, 0))

    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{wall } $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$\Pr_{wall} 0.71 $',rotation=0)
    else: 
        raise Exception('target name is not defined')



    fig2.savefig(os.path.join(output_path,'difference.pdf'),bbox_inches='tight',format='pdf')
    keras.utils.plot_model(model,to_file=os.path.join(output_path,"network.png"),show_shapes=True,dpi=100)
    
    return None





def stat_plots(mean_dataset_loc,batches):
    from DataHandling.features.stats import get_valdata
    import matplotlib.pyplot as plt
    import xarray as xr
    import numpy as np
    
    mean = xr.open_mfdataset(mean_dataset_loc, parallel=True)
    mean = mean.persist()
    mean = mean.groupby_bins("time", batches).mean()

    #Validation data
    val_u = get_valdata('u')

    linerRegion = np.linspace(0, 9)
    logRegion = 1 / 0.4 * np.log(np.linspace(20, 180)) + 5
    figScale = 2
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12 * figScale, 6 * figScale))
    ax1.plot('y+', 'u_plusmean', 'ok', data=val_u, label='DNS validation data')
    colorList = ['*b', '*c', '*y', '*g', '.b', '.c', '.y', '.g', 'vb', 'vc', 'vy', 'vg', '<b', '<c', '<y', '<g'] * 30
    # Plotting the batches in mean for U
    for i in range(len(mean.time_bins)):
        ax1.plot(mean.y_plus, mean.u_plusmean.isel(time_bins=i), colorList[i], label='DNS batch ' + str(i))

    ax1.plot(linerRegion, linerRegion, 'r', label='Linear Region')
    ax1.plot(np.linspace(20, 180), logRegion, 'm', linewidth=5, label='Log Region')

    ax1.set_title('Normalized mean values')
    ax1.set_xscale('log')
    ax1.set_xlabel('$y^{+}$')
    ax1.set_ylabel('$<u^{+}>$')
    ax1.set_xlim(1, 300)
    ax1.set_ylim(0, 20)
    ax1.minorticks_on()
    ax1.grid(True, which="both", linestyle='--')
    ax1.legend(prop={"size": 17})

    # Now for <u_rms>

    ax2.plot('y+', 'u_plusRMS', 'ok', data=val_u, label='DNS validation data')
    ax2.set_title('Normalized RMS of fluctuations')
    for i in range(len(mean.time_bins)):
        ax2.plot(mean.y_plus, mean.u_plusRMS.isel(time_bins=i), colorList[i], label='DNS batch ' + str(i))

    ax2.set_xscale('log')
    ax2.set_xlabel('$y^{+}$')
    ax2.set_ylabel("$u^{+}_{RMS}$")
    ax2.set_xlim(1, 300)
    ax2.minorticks_on()
    ax2.grid(True, which="both", linestyle='--')
    ax2.legend(prop={"size": 17})
    plt.tight_layout()
    plt.savefig("/home/au643300/DataHandling/reports/figures/u_val.pdf", bbox_inches='tight')

    a = get_valdata('pr1')
    b = get_valdata('pr71')
    c = get_valdata('pr0025')

    val_pr = [a, b, c]
    val_pr = val_pr[0].join(val_pr[1:])

    linerRegion = np.linspace(0, 9)
    logRegion = 1 / 0.43 * np.log(np.linspace(20, 180)) + 3
    figScale = 2

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12 * figScale, 6 * figScale))
    ax1.plot('pr1_y+', 'pr1_plusmean', 'ok', data=val_pr, label='DNS validation data Pr=1')
    ax1.plot('pr0.71_y+', 'pr0.71_plusmean', 'or', data=val_pr, label='DNS validation data Pr=0.71')
    ax1.plot('pr0.025_y+', 'pr0.025_plusmean', 'om', data=val_pr, label='DNS validation data Pr=0.025')
    colorList = ['*b', '*c', '*y', '*g', '.b', '.c', '.y', '.g', 'vb', 'vc', 'vy', 'vg', '<b', '<c', '<y', '<g'] * 30

    # Plotting the batches in mean for the different Pr
    pr_list = ['pr1', 'pr0.71', 'pr0.2', 'pr0.025']
    j = 0
    for i in range(len(mean.time_bins)):
        for Pr in pr_list:
            ax1.plot(mean.y_plus, mean[Pr + '_plusmean'].isel(time_bins=i), colorList[j * 4 + i],
                     label='DNS batch' + str(i) + 'Pr=' + Pr[2:])
            j = j + 1

    dualColor = ['k', 'r']
    j = 0
    for Pr in pr_list:
        ax1.plot(linerRegion, linerRegion * float(Pr[2:]), dualColor[i % 2],
                 label='Linear Region y+*Pr ' + 'Pr=' + Pr[2:])
        j = j + 1

    ax1.set_xscale('log')
    ax1.set_xlabel('$y^+$')
    ax1.set_ylabel(r'$<\theta^{+}>$')
    ax1.set_xlim(1, 300)
    ax1.set_ylim(0, 20)
    ax1.grid(True, which="both", linestyle='--')
    ax1.legend(loc='best', prop={"size": 15})

    # Now for <Pr_rms>

    ax2.plot('pr1_y+', 'pr1_plusRMS', 'ok', data=val_pr, label='DNS validation data Pr=1')
    ax2.plot('pr0.71_y+', 'pr0.71_plusRMS', 'or', data=val_pr, label='DNS validation data Pr=0.71')
    ax2.plot('pr0.025_y+', 'pr0.025_plusRMS', 'om', data=val_pr, label='DNS validation data Pr=0.025')
    ax2.set_title('Normalized RMS of fluctuations')
    j = 0
    for i in range(len(mean.time_bins)):
        for Pr in pr_list:
            ax2.plot(mean.y_plus, mean[Pr + '_plusRMS'].isel(time_bins=i), colorList[j * 4 + i],
                     label='DNS batch' + str(i) + 'Pr=' + Pr[2:])
            j = j + 1

    ax2.set_xscale('log')
    ax2.set_xlabel('$y^+$')
    ax2.set_ylabel(r'$\theta ^{+}_{RMS}$')
    ax2.set_xlim(1, 300)
    ax2.set_ylim(0, 3)
    ax2.grid(True, which="both", linestyle='--')
    ax2.legend(loc='best', prop={"size": 15})

    plt.tight_layout()
    plt.savefig("/home/au643300/DataHandling/reports/figures/Pr_val.pdf", bbox_inches='tight')

