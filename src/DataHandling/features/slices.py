
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


def load(data_loc,repeat=10,shuffle_size=100,batch_s=5):
      """A function that loads in a TFRecord from a saved location

      Args:
          data_loc (string): The TFRecords location
          repeat (int): How many repeats of the data
          shuffle_size (int): How big a shuffle cache should be
          batch_s (int): Size of each batch

      Returns:
          datase: tuple of the data 
      """
      #Her skal jeg implmntere det sidste så jeg faktisk får et dataset ud
      import tensorflow as tf
      
      
      dataset = tf.data.TFRecordDataset([data_loc],compression_type='GZIP')
      
      len_data=len(list(dataset))
      
      dataset=dataset.map(read_tfrecords)
      dataset=dataset.repeat(repeat)
      dataset=dataset.shuffle(buffer_size=shuffle_size)
      
      train_size = 20
      valid_size = 30
      test_size = 50

      train = dataset.take(train_size)
      remaining = dataset.skip(train_size)
      valid = remaining.take(valid_size)
      test = remaining.skip(valid_size)

      
      dataset_test=


      dataset=dataset.batch(batch_size=batch_s,num_parallel_calls=tf.data.AUTOTUNE)
      return dataset








def save(y_plus,data="/home/au643300/NOBACKUP/data/interim/data.zarr/"):
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
      import tensorflow.train as tft
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

      def custom_optimize(dsk, keys):
            dsk = dask.optimization.inline(dsk, inline_constants=True)
            return dask.array.optimization.optimize(dsk, keys)


      cluster=SLURMCluster(cores=8,
                        memory="50GB",
                        queue='q64',
                        walltime='0-01:00:00',
                        local_directory='/scratch/$SLURM_JOB_ID',
                        interface='ib0',
                        scheduler_options={'interface':'ib0'},
                        extra=["--lifetime", "50m"]
                        )


      def serialize(u_vel,tau_wall):
            u_vel_fea=tft.Feature(bytes_list=tft.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(u_vel)).numpy()]))
            tau_wall_fea=tft.Feature(bytes_list=tft.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(tau_wall)).numpy()]))

            features_dict={
                        'u_vel': u_vel_fea,
                        'tau_wall': tau_wall_fea
            }
            
            proto=tft.Example(features=tf.train.Features(feature=features_dict))
            return proto.SerializeToString()



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

      client=Client(cluster)
      cluster.adapt(minimum_jobs=0,maximum_jobs=4)

      

      with dask.config.set(array_optimize=custom_optimize):
            u_vel=dask.optimize(slice['u_vel'])[0]
            tau_wall=dask.optimize(slice['tau_wall'])[0]
            results=[u_vel,tau_wall]
            results=dask.optimize(results)[0]
            dask.compute(*results)
      

      save_loc="/home/au643300/DataHandling/data/processed"+"/y_plus_"+str(y_plus)

      #shuffle and save
  



      options = tf.io.TFRecordOptions(compression_type="GZIP")
      with tf.io.TFRecordWriter(save_loc,options) as writer:
            for i in range(u_vel.shape[0]):
                  test=serialize(u_vel[i,:,:],tau_wall[i,:,:])
                  writer.write(test)
    
      
      return None





