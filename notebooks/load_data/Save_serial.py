#!/usr/bin/env python
# coding: utf-8

# Så prøver vi en gang til at få det her shizzle til at virke. Først alle import

# In[1]:


import os
import sys


# In[2]:


os.chdir('/home/au643300/DataHandling/')


# Sætter Datahandling in i min syspath

# In[3]:


sys.path.append('/home/au643300/DataHandling/')


# In[4]:


from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed
import glob
from src.data.tonetCDF import to_netcdf
import xarray as xr


# Deleting the files smaller than 500 mb, as they are defect

# In[10]:


raw_path="/home/au643300/DataHandling/data/raw/"
files_raw=glob.glob(raw_path+'*.u')

interim_path="/home/au643300/NOBACKUP/interim/"
files_interim=glob.glob(interim_path+'*.nc')
for file in files_interim:
    size=os.path.getsize(file)
    if size < 500000000:
        print(file)
        
        #os.remove(file)
        


# Now finding the missing files
# 

# In[11]:


file_only_interim=[]
file_only_raw=[]
for file in files_interim:
    a=os.path.basename(file)
    a=a[:-3]
    file_only_interim.append(a)
for file in files_raw:
    b=os.path.basename(file)
    b=b[:-2]
    file_only_raw.append(b)




diff=list(set(file_only_interim)^set(file_only_raw))


# In[12]:


sym_path=[]
for name in diff:
    sym_path.append("/home/au643300/DataHandling/data/raw/"+name+'.u')

file_path=[]
for path in sym_path:
    file_path.append(os.readlink(path))




for file in file_path:
    ds=to_netcdf(file)
    ds.to_netcdf(interim_path + file[-12:-1]+ 'nc', engine='netcdf4')
    del ds


