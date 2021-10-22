



#%%
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from DataHandling.features import slices
import shutil


def get_plot_data(data):
    
   
    feature_list=[]
    target_list=[]

    for data_type in data:
        for i in data_type.take(1):
           feature_list.append(i[0])
           target_list.append(i[1].numpy())

    
    names=['train','validation','test']
    return feature_list,target_list,names


#%%

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


    #Plot of test, train, validation
    for i in range(3):
        plt.figure()
        #TODO Lavet et plot med features som ser lækkert ud
        #TODO sat axer på plottet
        #TODO lavet labels til colorbar
        fig, axes=plt.subplots(1,2,sharex=True,sharey=True)
        cbar_ax = fig.add_axes([.91, 0.25, .04, .5])
        axes[0].set_title('Target')
        
        sns.heatmap(target_list[i][1,:,:],ax=axes[0],square=True,xticklabels=False,yticklabels=False,cmap="rocket",cbar_ax=cbar_ax)
        axes[1].set_title('Prediction')
        sns.heatmap(predctions[i][1,:,:],ax=axes[1],square=True,xticklabels=False,yticklabels=False,cmap="rocket",cbar_ax=cbar_ax)
        # for feature in feature_list[i]:
        #     axes[2].set_title('Feature ' + feature)
        #     feature_var=feature_list[i][feature].numpy()
        #     sns.heatmap(feature_var[1,:,:] ,ax=axes[2],square=True,xticklabels=True,yticklabels=True,cmap="rocket")
        

        fig.tight_layout(rect=[0, 0, .9, 1])
        # plt.savefig(os.path.join(output_path,names[i]+".pdf"),dpi=100,bbox_inches='tight',format='pdf')

        # plt.figure()
        # sns.heatmap(predctions[i][1,:,:]-target_list[i][1,:,:],square=True,xticklabels=False,yticklabels=False,cmap="icefire")
        # plt.savefig(os.path.join(output_path,names[i]+"_difference"+".pdf"),dpi=100,bbox_inches='tight',format='pdf')


    plt.figure()
    keras.utils.plot_model(model,to_file=os.path.join(output_path,"network.png"),show_shapes=True,dpi=100)
    return None

var=['u_vel']
target=['tau_wall']
normalize=True
y_plus=15
model_name="sweet-firefly-6"

heatmap(model_name,var,target,y_plus,normalize)


