#%%
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


data="/home/au643300/NOBACKUP/data/interim/data.zarr/"
y_plus=15

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

#%%
#For now only u and tau are saved

client=Client(cluster)
cluster.adapt(minimum_jobs=0,maximum_jobs=5)



#with dask.config.set(array_optimize=custom_optimize):
u_vel=dask.optimize(slice['u_vel'])[0]
tau_wall=dask.optimize(slice['tau_wall'])[0]
results=[u_vel,tau_wall]
results=dask.optimize(results)[0]
results=dask.compute(*results)

u_vel=results[0].values
tau_wall=results[1].values

#%%

#Her tager jeg og deler dataen op i 3 forskellige dele og shuffler det hele rundt også


test_split=0.1
validation_split=0.2


num_snapshots=u_vel.shape[0]

train=np.arange(0,num_snapshots)


validation=np.random.choice(train,size=int(num_snapshots*validation_split),replace=False)
train=np.setdiff1d(train,validation)

test=np.random.choice(train,size=int(num_snapshots*test_split),replace=False)
train=np.setdiff1d(train,test)


np.random.shuffle(train)

#%%


save_loc="/home/au643300/DataHandling/data/processed"+"/y_plus_"+str(y_plus)


#TODO ændret test navnet her så det hele ikke fucker op

options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter(save_loc+"_train",options) as writer:
    for i in train:
            test=serialize(u_vel[i,:,:],tau_wall[i,:,:])
            writer.write(test)
    writer.close()


with tf.io.TFRecordWriter(save_loc+"_test",options) as writer:
    for i in test:
            test=serialize(u_vel[i,:,:],tau_wall[i,:,:])
            writer.write(test)
    writer.close()

with tf.io.TFRecordWriter(save_loc+"_validation",options) as writer:
    for i in validation:
            test=serialize(u_vel[i,:,:],tau_wall[i,:,:])
            writer.write(test)
    writer.close()