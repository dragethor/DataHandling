#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import os
import sys


# In[2]:


#os.chdir('/home/au643300/DataHandling/')
#sys.path.append('/home/au643300/DataHandling/')
os.environ['HDF5_USE_FILE_LOCKING']="FALSE"


# In[3]:


from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed,wait,fire_and_forget,progress
import glob
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import importlib
import dask
import zarr
import time
from rechunker import rechunk


# In[4]:


def custom_optimize(dsk, keys):
    dsk = dask.optimization.inline(dsk, inline_constants=True)
    return dask.array.optimization.optimize(dsk, keys)
sns.set_theme()
dask.config.set({"distributed.comm.timeouts.tcp": "50s"})
dask.config.set({'distributed.comm.retry.count':3})
dask.config.set({'distributed.comm.timeouts.connect':'25s'})
dask.config.set({"distributed.worker.use-file-locking":False})
zarr.blosc.use_threads = False


# Loads the Datahandling package(which is everything in the src folder)

# In[5]:


get_ipython().run_line_magic('load_ext', 'autoreload')


# In[6]:


get_ipython().run_line_magic('autoreload', '2')


# In[7]:


import importlib
from DataHandling.data.toxarr import to_xarr


# In[8]:


cluster=SLURMCluster(cores=8,
                     memory="31GiB",
                     processes=2,
                     queue='q64',
                     walltime='0-01:00:00',
                     local_directory='/scratch/$SLURM_JOB_ID',
                     interface='ib0',
                     scheduler_options={'interface':'ib0'},
                     extra=['--resources save=1',"--lifetime", "50m", "--lifetime-stagger", "4m"]
                    )
                    


# In[9]:


client=Client(cluster)


# In[10]:


cluster.adapt(minimum=2,maximum=8)


# In[11]:


client


# In[37]:


# test="/home/au643300/NOBACKUP/interim/"
# files=glob.glob(test+'*.nc')
# files=sorted(files)
# ex=xr.open_mfdataset(files[0:2],parallel=True,combine='nested',concat_dim='time')
# ex.attrs['field']=files[1][-7:-3]
# ex.to_zarr("/home/au643300/NOBACKUP/data.zarr")


# In[38]:


ex=xr.open_zarr("/home/au643300/NOBACKUP/data.zarr")
ex


# In[39]:


raw="/home/au643300/DataHandling/data/raw/"
files=glob.glob(raw+'*.u')
files=sorted(files)
file_names=[os.path.basename(path) for path in files]
file_names=[file[0:-2] for file in file_names]


# In[40]:


field='field.'+ex.attrs['field']
field


# In[41]:


start=[field==file for file in file_names]
index=int(np.where(start)[0])
index+=1
index


# In[43]:


for file in files[index:]:
    data=to_xarr(file)
    data.attrs['field']=file[-6:-2]
    data.to_zarr("/home/au643300/NOBACKUP/data.zarr", append_dim="time",compute=True)
    del data


# In[35]:


ds=xr.open_zarr("/home/au643300/NOBACKUP/data.zarr")
ds

