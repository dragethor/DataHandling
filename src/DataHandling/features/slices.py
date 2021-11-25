


import tensorflow as tf


def feature_description(save_loc):
    """Loads the json file descriping the file format for parsing the tfRecords

    Args:
        save_loc (string): The file path to the folder where the data is saved

    Returns:
        dict: dict used to read tfRecords
    """
    import os
    import tensorflow as tf
    import json
    feature_format={}

    with open(os.path.join(save_loc,"format.json"),'r') as openfile:
        format_json=json.load(openfile)
    


    for key in list(format_json.keys()):
        if format_json[key] =="array_serial":
            feature_format[key]= tf.io.FixedLenFeature([], tf.string, default_value="")
        else:
            print("other features than array has not yet been implemented!")
    return feature_format



def read_tfrecords(serial_data,format,target):
    """Reads the tfRecords and converts them to a tuple where first entry is the features and the second is the targets

    Args:
        serial_data (TFrecordDataset): The output of the function tf.data.TFRecordDataset
        format (dict): dict used to parse the TFrecord example format

    Returns:
        tuple: tuple of (features,labels)
    """
    import tensorflow as tf
      
    features=tf.io.parse_single_example(serial_data, format)

    dict_for_dataset={}

    for key, value in features.items():
        if value.dtype == tf.string:
            dict_for_dataset[key]=tf.io.parse_tensor(value,tf.float64)
        else:
            print("only arrays have been implemented")
    

    target_array=dict_for_dataset[target[0]]
    dict_for_dataset.pop(target[0])

     
    return (dict_for_dataset,target_array)






def load_from_scratch(y_plus,var,target,normalized,repeat=10,shuffle_size=100,batch_s=10):
    """Copyes TFrecord to scratch and loads the data from there

    Args:
        y_plus (int): the y_plus value to load data from
        var (list): list of features
        target (list): list of targets. only 1 for now
        repeat (int, optional): number of repeats of the dataset for each epoch. Defaults to 10.
        shuffle_size (int, optional): the size of the shuffle buffer. Defaults to 100.
        batch_s (int, optional): the number of snapshots in each buffer. Defaults to 10.

    Returns:
        [type]: [description]
    """
      
    import tensorflow as tf
    import os
    import shutil
    import xarray as xr
    save_loc=slice_loc(y_plus,var,target,normalized)

    if not os.path.exists(save_loc):
        raise Exception("data does not exist. Make som new")


    #copying the data to scratch
    scratch=os.path.join('/scratch/', os.environ['SLURM_JOB_ID'])
    #shutil.copytree(save_loc,scratch)
    #print("copying data to scratch")

    features_dict=feature_description(save_loc)


    
    splits=['train','validation','test']

    data=[]
    for name in splits:
        data_loc=os.path.join(scratch,name)
        shutil.copy2(os.path.join(save_loc,name),data_loc)
        dataset = tf.data.TFRecordDataset([data_loc],compression_type='GZIP',buffer_size=100,num_parallel_reads=tf.data.experimental.AUTOTUNE)
        dataset=dataset.map(lambda x: read_tfrecords(x,features_dict,target),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset=dataset.shuffle(buffer_size=shuffle_size)
        dataset=dataset.repeat(repeat)
        dataset=dataset.batch(batch_size=batch_s)
        dataset=dataset.prefetch(3)
        data.append(dataset)
    return data



def load_validation(y_plus,var,target,normalized):

      
    import tensorflow as tf
    import os
    import shutil
    
    save_loc=slice_loc(y_plus,var,target,normalized)

    if not os.path.exists(save_loc):
        raise Exception("data does not exist. Make som new")


    #copying the data to scratch
    scratch=os.path.join('/scratch/', os.environ['SLURM_JOB_ID'])
    #shutil.copytree(save_loc,scratch)
    #print("copying data to scratch")

    features_dict=feature_description(save_loc)


    data_unorder=[]

    #validation
    data_loc=os.path.join(scratch,'validation')
    shutil.copy2(os.path.join(save_loc,'validation'),data_loc)
    dataset = tf.data.TFRecordDataset([data_loc],compression_type='GZIP',buffer_size=100,num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset=dataset.map(lambda x: read_tfrecords(x,features_dict,target),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset=dataset.take(-1)
    dataset=dataset.cache()
    len_data_val=len(list(dataset))
    dataset=dataset.batch(len_data_val)
    dataset=dataset.get_single_element()
    data_unorder.append(dataset)

    #train
    data_loc=os.path.join(scratch,'train')
    shutil.copy2(os.path.join(save_loc,'train'),data_loc)
    dataset = tf.data.TFRecordDataset([data_loc],compression_type='GZIP',buffer_size=100,num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset=dataset.map(lambda x: read_tfrecords(x,features_dict,target),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset=dataset.take(len_data_val)
    dataset=dataset.batch(len_data_val)
    dataset=dataset.get_single_element()
    data_unorder.append(dataset)    
    
    
    
    #test
    data_loc=os.path.join(scratch,'test')
    shutil.copy2(os.path.join(save_loc,'test'),data_loc)
    dataset = tf.data.TFRecordDataset([data_loc],compression_type='GZIP',buffer_size=100,num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset=dataset.map(lambda x: read_tfrecords(x,features_dict,target),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset=dataset.take(-1)
    dataset=dataset.cache()
    len_data_test=len(list(dataset))
    dataset=dataset.batch(len_data_test)
    dataset=dataset.get_single_element()
    data_unorder.append(dataset)

    data=[data_unorder[1],data_unorder[0],data_unorder[2]]

    return data
    



def save_tf(y_plus,var,target,data,normalized=False):
    """Takes a xarray dataset extracts the variables in var and saves them as a tfrecord

    Args:
        y_plus (int): at which y_plus to take a slice
        var (list): list of inputs to save. NOT with target
        target (list): list of target. Only 1 target for now
        data (xarray): dataset of type xarray
        normalized(bool): if the data is normalized or not

    Returns:
        None:
    """

    import os
    import xarray as xr
    import numpy as np
    import dask
    import tensorflow as tf
    from DataHandling import utility
    import shutil
    import json

    def custom_optimize(dsk, keys):
        dsk = dask.optimization.inline(dsk, inline_constants=True)
        return dask.array.optimization.optimize(dsk, keys)



    def numpy_to_feature(numpy_array):
        """Takes an numpy array and returns a tf feature

        Args:
            numpy_array (ndarray): numpy array to convert to tf feature

        Returns:
            Feature: Feature object to use in an tf example
        """
        feature=tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(numpy_array)).numpy()]))
        return feature



    def serialize(slice_array,var):
        """Constructs an serialzied tf.Example package

        Args:
            slice_array (xarray): A xaray
            var (list): a list of the variables that are to be serialized

        Returns:
            protostring: protostring of tf.train.Example
        """

        feature_dict={}
        for name in var:
            feature=slice_array[name].values
            if type(feature) is np.ndarray:
                feature_dict[name] = numpy_to_feature(feature)
            else:
                raise Exception("other inputs that xarray/ numpy has not yet been defined")
        
        proto=tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return proto.SerializeToString()


    def split_test_train_val(slice_array,test_split=0.1,validation_split=0.2):
        """Splits the data into train,test,val

        Args:
            slice_array (xarray): The sliced data to be split
            test_split (float, optional): the test split. Defaults to 0.1.
            validation_split (float, optional): the validation split. Defaults to 0.2.

        Returns:
            tuple: returns the selected indices for the train, validation,test split
        """
        num_snapshots=len(slice_array['time'])
        train=np.arange(0,num_snapshots)
        validation=np.random.choice(train,size=int(num_snapshots*validation_split),replace=False)
        train=np.setdiff1d(train,validation)
        test=np.random.choice(train,size=int(num_snapshots*test_split),replace=False)
        train=np.setdiff1d(train,test)
        np.random.shuffle(train)

        return train, validation, test



    def save_load_dict(var,save_loc):
        """Saves an json file with the file format. Makes it possible to read the data back again

        Args:
            var (list): list of variables to include
        """
        load_dict={}

        for name in var:
            load_dict[name] = "array_serial"

        with open(os.path.join(save_loc,'format.json'), 'w') as outfile:
            json.dump(load_dict,outfile)


    client, cluster =utility.slurm_q64(1,time='0-03:00:00',ram='100GB')

    save_loc=slice_loc(y_plus,var,target,normalized)
    
    #append the target
    var.append(target[0])


    #select y_plus value and remove unessary components. Normalize if needed

    slice_array=data
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity


    if target[0]=='tau_wall':
        target_slice1=slice_array['u_vel'].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")
        target_slice1=nu*target_slice1
        
        #target_slice2=slice_array['u_vel'].differentiate('y').sel(y=slice_array['y'].max(),method="nearest")
        #target_slice2=nu*target_slice2
        
        if normalized==True:
            target_slice1=(target_slice1-target_slice1.mean(dim=('time','x','z')))/(target_slice1.std(dim=('time','x','z')))
    
    #Checking if the target contains _flux
    elif target[0][-5:] =='_flux':
        target_slice1=slice_array[target[0][:-5]].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")
        pr_number=float(target[0][2:-5])
        target_slice1=nu/(pr_number)*target_slice1
        
        #target_slice2=slice_array[target[0][:-5]].differentiate('y').sel(y=slice_array['y'].max(),method="nearest")
        #target_slice2=nu/(pr_number)*target_slice2
        
        if normalized==True:
            target_slice1=(target_slice1-target_slice1.mean(dim=('time','x','z')))/(target_slice1.std(dim=('time','x','z')))
    else:
        target_slice1=slice_array[target[0]].sel(y=utility.y_plus_to_y(0),method="nearest")

        #target_slice2=slice_array[target[0]].sel(y=slice_array['y'].max(),method="nearest")
        if normalized==True:
            target_slice1=(target_slice1-target_slice1.mean(dim=('time','x','z')))/(target_slice1.std(dim=('time','x','z')))


    other_wall_y_plus=utility.y_to_y_plus(slice_array['y'].max())-y_plus
    
    if normalized==True:
        slice_array=(slice_array-slice_array.mean(dim=('time','x','z')))/(slice_array.std(dim=('time','x','z')))

    
    

    wall_1=slice_array.sel(y=utility.y_plus_to_y(y_plus),method="nearest")
    wall_1[target[0]]=target_slice1
    wall_1=wall_1[var]

    #wall_2=slice_array.sel(y=utility.y_plus_to_y(other_wall_y_plus),method="nearest")
    #wall_2[target[0]]=target_slice2
    #wall_2=wall_2[var]
    
 
    #wall_1,wall_2=dask.compute(*[wall_1,wall_2])

    #shuffle the data, split into 3 parts and save
    train_1, validation_1, test_1 = split_test_train_val(wall_1)

    wall_1=wall_1.compute()
    

    #train_2, validation_2, test_2 = split_test_train_val(wall_2)

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    else:
        print('deleting old version')
        shutil.rmtree(save_loc)           
        os.makedirs(save_loc)


    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(os.path.join(save_loc,'train'),options) as writer:
        for i in train_1:
            write_d=serialize(wall_1.isel(time=i),var)
            writer.write(write_d)
        # for i in train_2:
        #         write_d=serialize(wall_2.isel(time=i),var)
        #         writer.write(write_d)
        writer.close()


    with tf.io.TFRecordWriter(os.path.join(save_loc,'test'),options) as writer:
        for i in test_1:
            write_d=serialize(wall_1.isel(time=i),var)
            writer.write(write_d)
        # for i in test_2:
        #         write_d=serialize(wall_2.isel(time=i),var)
        #         writer.write(write_d)
        writer.close()

    with tf.io.TFRecordWriter(os.path.join(save_loc,'validation'),options) as writer:
        for i in validation_1:
            write_d=serialize(wall_1.isel(time=i),var)
            writer.write(write_d)
        # for i in validation_2:
        #         write_d=serialize(wall_2.isel(time=i),var)
        #         writer.write(write_d)    
        writer.close()


    save_load_dict(var,save_loc)
    client.close()
    return None


def slice_loc(y_plus,var,target,normalized):
    """where to save the slices

    Args:
        y_plus (int): y_plus value of slice
        var (list): list of variables
        target (list): list of targets
        normalized (bool): if the data is normalized or not

    Returns:
        str: string of file save location
    """
    import os
    if normalized==True:
        slice_loc=os.path.join("/home/au643300/DataHandling/data/processed",'y_plus_'+str(y_plus)+"_var"+str(len(var))+"_"+str(target[0])+"_normalized")
    else:
        slice_loc=os.path.join("/home/au643300/DataHandling/data/processed",'y_plus_'+str(y_plus)+"_var"+str(len(var))+"_"+str(target[0]))

    return slice_loc