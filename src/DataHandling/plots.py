

def relative_error(data,model_name):
    """Finds the relative error from a batch of snapshots

    Args:
        data (TFrecord): TFrecord data
        model_name (str): name of the run. In the form of a wandb name

    Returns:
        list: a list of the error in percent for train, validation, test data
    """

    from DataHandling import utility
    import numpy as np
    from tensorflow import keras
    import os

    feature_list=[]
    target_list=[]

    for data_type in data:
        for i in data_type.take(1):
            feature_list.append(i[0])
            target_list.append(i[1].numpy())

    feature_list,target_list,names=utility.get_data(data)

    model_path=os.path.join("/home/au643300/DataHandling/models/trained/",model_name)
    model=keras.models.load_model(model_path)

    predctions=[]
    for features in feature_list:
        predctions.append(model.predict(features))

    predctions=[np.squeeze(x,axis=3) for x in predctions]

    error=[]

    for i in range(3):
        err=(predctions[i]-target_list[i])/target_list[i]*100
        error.append(np.mean(err))
    return error



def heatmap(model_name,var,target,y_plus,normalize):
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorflow import keras
    from DataHandling.features import slices
    import shutil



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

    feature_list,target_list,names=get_plot_data(data)



    predctions=[]
    for features in feature_list:
        predctions.append(model.predict(features))

    predctions=[np.squeeze(x,axis=3) for x in predctions]



    #%%


    #Plot of test, train, validation
    for i in range(3):
        plt.figure()
        #TODO hvis der er mere end 1 feature skal der laves noget smart med subplots her
        fig, axes=plt.subplots(1,3,sharex=True,sharey=True)
        #cbar_ax = fig.add_axes([.91, 0.2, .04, .5])
        axes[0].set_title('Target')
        sns.heatmap(target_list[i][1,:,:],ax=axes[0],square=True,xticklabels=False,yticklabels=False,cmap="rocket")
        axes[1].set_title('Prediction')
        sns.heatmap(predctions[i][1,:,:],ax=axes[1],square=True,xticklabels=False,yticklabels=False,cmap="rocket")
        for feature in feature_list[i]:
            axes[2].set_title('Feature ' + feature)
            feature_var=feature_list[i][feature].numpy()
            sns.heatmap(feature_var[1,:,:] ,ax=axes[2],square=True,xticklabels=False,yticklabels=False,cmap="rocket")
        

        #fig.tight_layout(rect=[0, 0, .9, 1])
        plt.savefig(os.path.join(output_path,names[i]+".pdf"),dpi=100,bbox_inches='tight',format='pdf')

        plt.figure()
        sns.heatmap(predctions[i][1,:,:]-target_list[i][1,:,:],square=True,xticklabels=False,yticklabels=False,cmap="icefire")
        plt.savefig(os.path.join(output_path,names[i]+"_difference"+".pdf"),dpi=100,bbox_inches='tight',format='pdf')


    plt.figure()
    keras.utils.plot_model(model,to_file=os.path.join(output_path,"network.png"),show_shapes=True,dpi=100)
    return None




def stat_plots(mean_dataset_loc,batches):
    from DataHandling.features.read_valdata import get_valdata
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


def get_plot_data(data):
    
   
    feature_list=[]
    target_list=[]

    for data_type in data:
        for i in data_type.take(1):
           feature_list.append(i[0])
           target_list.append(i[1].numpy())

    
    names=['train','validation','test']
    return feature_list,target_list,names




def heatmap(model_name,var,target,y_plus,normalize):
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorflow import keras
    from DataHandling.features import slices
    import shutil



    model_path=os.path.join("/home/au643300/DataHandling/models/trained/",model_name)
    output_path='/home/au643300/DataHandling/reports/figures'
    data_path=slices.slice_loc(y_plus,var,target,normalize)+"/"
    data_folder=os.path.basename(os.path.dirname(data_path))

    output_path=os.path.join(output_path,model_name+'_'+data_folder)

    data=slices.load_from_scratch(y_plus,var,target,normalized=normalize)
    
    #TODO ændret det her sådan at den enten bruger h5 eller folder
    model=keras.models.load_model(os.path.join(model_path,"model.h5"))



    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        print('deleting old version')
        shutil.rmtree(output_path)           
        os.makedirs(output_path)

    feature_list,target_list,names=get_plot_data(data)



    predctions=[]
    for features in feature_list:
        predctions.append(model.predict(features))

    predctions=[np.squeeze(x,axis=3) for x in predctions]



    #%%


    #Plot of test, train, validation
    for i in range(3):
        plt.figure()
        fig, axes=plt.subplots(1,2,sharex=True,sharey=True)
        cbar_ax = fig.add_axes([.91, 0.2, .04, .5])
        axes[0].set_title('Target')
        sns.heatmap(target_list[i][1,:,:],ax=axes[0],square=True,xticklabels=False,yticklabels=False,cmap="rocket",cbar_ax=cbar_ax)
        axes[1].set_title('Prediction')
        sns.heatmap(predctions[i][1,:,:],ax=axes[1],square=True,xticklabels=False,yticklabels=False,cmap="rocket",cbar_ax=cbar_ax)
        fig.tight_layout(rect=[0, 0, .9, 1])
        plt.savefig(os.path.join(output_path,names[i]+".pdf"),dpi=200,bbox_inches='tight',format='pdf')

        plt.figure()
        sns.heatmap(predctions[i][1,:,:]-target_list[i][1,:,:],square=True,xticklabels=False,yticklabels=False,cmap="icefire")
        plt.savefig(os.path.join(output_path,names[i]+"_difference"+".pdf"),dpi=200,bbox_inches='tight',format='pdf')


    plt.figure()
    keras.utils.plot_model(model,to_file=os.path.join(output_path,"network.png"),show_shapes=True,dpi=200)
    return None