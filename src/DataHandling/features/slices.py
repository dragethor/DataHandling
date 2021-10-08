








def read_tfrecords(serial_data):
      """reads tfrecords from file and unserializeses them

      Args:
          serial_data (TFrecord): [Tfrecord that needs to be unserialzed]

      Returns:
          (u_vel,tau_wall) (tuple): A tuple of u_vel and tau_wall
      """
      import tensorflow as tf
      
      format = {
      "u_vel": tf.io.FixedLenFeature([], tf.string, default_value=""),
      "tau_wall": tf.io.FixedLenFeature([], tf.string, default_value="")
      }

      
      features=tf.io.parse_single_example(serial_data, format)

      u_vel=tf.io.parse_tensor(features['u_vel'],tf.float64)
      tau_wall=tf.io.parse_tensor(features['tau_wall'],tf.float64)
      return (u_vel, tau_wall)




def load_from_scratch(y_plus,var,repeat=10,shuffle_size=100,batch_s=10):
      """A function that loads in a TFRecord from a saved location

      Args:
          y_plus (int): The y plus value 
          repeat (int): How many repeats of the data
          shuffle_size (int): How big a shuffle cache should be
          batch_s (int): Size of each batch

      Returns:
          data: data[0]=test, data[1]=train, data[2]=validation 
      """
      
      import tensorflow as tf
      import os
      import shutil

      save_loc=os.path.join("/home/au643300/DataHandling/data/processed",*('y_plus_'+str(y_plus)))

      if not os.path.exists(save_loc):
            print("data does not exist. Making new",flush=True)
            save_tf(y_plus,var)
      
      
      
      #copying the data to scratch
      scratch=os.path.join('/scratch/', os.environ['SLURM_JOB_ID'])
      shutil.copytree(save_loc,scratch)
      


      splits=['test','train','validation']

      data=[]
      for name in splits:
            data_loc=os.path.join(save_loc,name)
            dataset = tf.data.TFRecordDataset([data_loc],compression_type='GZIP')
            dataset=dataset.map(read_tfrecords)
            dataset=dataset.shuffle(buffer_size=shuffle_size)
            dataset=dataset.repeat(repeat)
            dataset=dataset.batch(batch_size=batch_s)
            dataset.prefetch(tf.data.experimental.AUTOTUNE)
            data=data.append(dataset)
      return data[0],data[1],data[2]



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

    client, cluster =utility.slurm_q64(1,ram='75GB')


    #tau_wall is allways used
    var.append('tau_wall')




    slice_array=data
    slice_array=slice_array.assign(tau_wall=slice_array['u_vel'].differentiate('y')).sel(y=utility.y_plus_to_y(0),method="nearest")
    slice_array=slice_array.sel(y=utility.y_plus_to_y(y_plus), method="nearest")


    slice_array=slice_array[var]
    slice_array=dask.compute(slice_array)
    #slice_array=slice_array.result()


    save_loc=os.path.join("/home/au643300/DataHandling/data/processed",'y_plus_'+str(y_plus))+"_var_"+str(len(var)-1)


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
    client.close()
    return None


def save_legacy(y_plus,data="/home/au643300/NOBACKUP/data/interim/data.zarr/"):
      """Takes the full dataset and saves slices of tau_wall and u_vel at some y+ value

      Args:
          y_plus (int): The chosen y+ value
          data (str, optional): Data location. Defaults to "/home/au643300/NOBACKUP/data/interim/data.zarr/".

      Returns:
          None: 
      """
      import os
      from dask_jobqueue import SLURMCluster
      from dask.distributed import Client
      import xarray as xr
      import numpy as np
      import dask
      import zarr
      import time
      import tensorflow as tf
      from DataHandling import utiliy

      def custom_optimize(dsk, keys):
            dsk = dask.optimization.inline(dsk, inline_constants=True)
            return dask.array.optimization.optimize(dsk, keys)


      def serialize(u_vel,tau_wall):
            u_vel_fea=tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(u_vel)).numpy()]))
            tau_wall_fea=tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(tau_wall)).numpy()]))

            features_dict={
                        'u_vel': u_vel_fea,
                        'tau_wall': tau_wall_fea
            }
            
            proto=tf.train.Example(features=tf.train.Features(feature=features_dict))
            return proto.SerializeToString()

      client=utiliy.slurm_q64(maximum_jobs=6)

      Re_Tau = 395 #Direct from simulation
      Re = 10400 #Direct from simulation
      nu = 1/Re #Kinematic viscosity
      u_tau = Re_Tau*nu

      #converts between y_plus and y
      y_func= lambda y_plus : y_plus*nu/u_tau

      #Opening up the full dataset
      source=xr.open_zarr(data)

      slice=source
      slice=slice.assign(tau_wall=slice['u_vel'].differentiate('y').isel(y=-1))
      slice=slice.sel(y=y_func(15), method="nearest")


      #For now only u and tau are saved

      
      u_vel=slice['u_vel']
      tau_wall=slice['tau_wall']
      results=[u_vel,tau_wall]
      results=dask.optimize(results)[0]
      results=dask.compute(*results)
      u_vel=results[0].values
      tau_wall=results[1].values
      

      save_loc=os.path.join("/home/au643300/DataHandling/data/processed",'y_plus_'+str(y_plus))
      #shuffle the data, split into 3 parts and save and save
  
      test_split=0.1
      validation_split=0.2



      num_snapshots=u_vel.shape[0]

      train=np.arange(0,num_snapshots)


      validation=np.random.choice(train,size=int(num_snapshots*validation_split),replace=False)
      train=np.setdiff1d(train,validation)

      test=np.random.choice(train,size=int(num_snapshots*test_split),replace=False)
      train=np.setdiff1d(train,test)


      np.random.shuffle(train)



      options = tf.io.TFRecordOptions(compression_type="GZIP")
      with tf.io.TFRecordWriter(os.path.join(save_loc,"train"),options) as writer:
            for i in train:
                        write_d=serialize(u_vel[i,:,:],tau_wall[i,:,:])
                        writer.write(write_d)
            writer.close()


      with tf.io.TFRecordWriter(os.path.join(save_loc,"test"),options) as writer:
            for i in test:
                        write_d=serialize(u_vel[i,:,:],tau_wall[i,:,:])
                        writer.write(write_d)
            writer.close()

      with tf.io.TFRecordWriter(os.path.join(save_loc,"validation"),options) as writer:
            for i in validation:
                        write_d=serialize(u_vel[i,:,:],tau_wall[i,:,:])
                        writer.write(write_d)
            writer.close()
    
      
      return None





