




#%%

import os
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import xarray as xr
import numpy as np
import dask
import zarr
import time
import tensorflow as tf
from DataHandling import utility
from DataHandling.data_raw import make_dataset

#%%



data=make_dataset.to_xarr("/home/au643300/DataHandling/data/raw/field.0496.u")

#%%
dicto={
    "u_vel":
}

#%%



def custom_optimize(dsk, keys):
    dsk = dask.optimization.inline(dsk, inline_constants=True)
    return dask.array.optimization.optimize(dsk, keys)



def serialize_numpy(numpy_array):
    serial=tf.io.serialize_tensor(tf.convert_to_tensor(numpy_array)).numpy()
    return serial



var=['u_vel','v_vel']

ds=data[var]

ds1=ds.to_dict(data=True)

ds2=ds1['data_vars']


def tf_example(var_list):
    #skal allarede være computed og sliced
    features_dict={}
    for var_name in var_list:
        features_dict[var_name]=data



    





def serialize(u_vel,tau_wall):
    u_vel_fea=tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(u_vel)).numpy()]))
    tau_wall_fea=tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(tau_wall)).numpy()]))

    features_dict={
                'u_vel': u_vel_fea,
                'tau_wall': tau_wall_fea
    }
    
    proto=tf.train.Example(features=tf.train.Features(feature=features_dict))
    return proto.SerializeToString()



#%%

client=utility.slurm_q64(maximum_jobs=6)

#%%
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
