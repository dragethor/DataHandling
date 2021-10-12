


#%%


def save_tf(y_plus,var,data):
    """Takes a xarray dataset extracts the variables in var and saves them as a tfrecord

    Args:
        y_plus (int): at which y_plus to take a slice
        var (list): list of inputs to save. NOT with target tau_wall
        data (xarray): dataset of type xarray

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

    client, cluster =utility.slurm_q64(2)



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
                feature_dict[name] = numpy_to_feature(slice_array[name].values)
            else:
                print("other inputs that xarray/ numpy has not yet been defined")
        
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


    #%%
    #for name in var:
            #load_dict[name] = tf.io.FixedLenFeature([], tf.string, default_value="")




    #tau_wall is allways used
    var.append('tau_wall')




    slice_array=data
    slice_array=slice_array.assign(tau_wall=slice_array['u_vel'].differentiate('y').sel(y=utility.y_plus_to_y(0)))
    slice_array=slice_array.sel(y=utility.y_plus_to_y(y_plus), method="nearest")


    slice_array=slice_array[var]
    slice_array=dask.optimize(slice_array)[0]
    slice_array=slice_array.compute()


    save_loc=os.path.join("/home/au643300/DataHandling/data/processed",'y_plus_'+str(y_plus))


    #shuffle the data, split into 3 parts and save
    train, validation, test = split_test_train_val(slice_array)

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    else:
        print('deleting old version')
        shutil.rmtree(save_loc)           
        os.makedirs(save_loc)


    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(os.path.join(save_loc,'train'),options) as writer:
        for i in train:
                    write_d=serialize(slice_array.isel(time=i),var)
                    writer.write(write_d)
        writer.close()


    with tf.io.TFRecordWriter(os.path.join(save_loc,'test'),options) as writer:
        for i in test:
                    write_d=serialize(slice_array.isel(time=i),var)
                    writer.write(write_d)
        writer.close()

    with tf.io.TFRecordWriter(os.path.join(save_loc,'validation'),options) as writer:
        for i in validation:
                    write_d=serialize(slice_array.isel(time=i),var)
                    writer.write(write_d)
        writer.close()


    save_load_dict(var,save_loc)

    return None



