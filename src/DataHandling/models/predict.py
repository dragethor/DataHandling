def predict(model_name,overwrite,model,y_plus,var,target,normalized):
    import os
    from DataHandling import utility
    from DataHandling.features import slices
    import shutil
    import numpy as np
    
    _,output_path=utility.model_output_paths(model_name,y_plus,var,target,normalized)


    if not os.path.exists(output_path):

        os.makedirs(output_path)

    elif os.path.exists(os.path.join(output_path,'targets.npz')) and overwrite==False:

        raise Exception("Data exists and overwrite is set to false. Exiting")

    elif os.path.exists(os.path.join(output_path,'targets.npz')) and overwrite==True:

        print("deleting folder",flush=True)
        shutil.rmtree(output_path)




    data=slices.load_validation(y_plus,var,target,normalized)


    feature_list=[]

    target_list=[]


    for data_type in data:

        feature_list.append(data_type[0])

        target_list.append(data_type[1].numpy())


    predctions=[]


    predctions.append(model.predict(feature_list[0]))
    predctions.append(model.predict(feature_list[1]))
    predctions.append(model.predict(feature_list[2]))

    predctions=[np.squeeze(x,axis=3) for x in predctions]

    np.savez_compressed(os.path.join(output_path,"predictions"),train=predctions[0],val=predctions[1],test=predctions[2])
    np.savez_compressed(os.path.join(output_path,"targets"),train=target_list[0],val=target_list[1],test=target_list[2])


    print("Saved data",flush=True)