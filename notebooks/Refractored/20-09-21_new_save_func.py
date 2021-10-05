

#%%
import glob
import os
import numpy as np
import xarray as xr

raw = "/home/au643300/DataHandling/data/raw/"

files = glob.glob(raw + '*.u')
files = sorted(files)
file_names = [os.path.basename(path) for path in files]
file_names = [file[0:-2] for file in file_names]



if not os.path.exists(store):
    print("Making new zarr array",flush=True)
    data = to_xarr(files[0])
    data.attrs['field'] = file_names[0]
    data.to_zarr(store, compute=True)
    print("saved "+file_names[0],flush=True)
    del data




ex = xr.open_zarr(store)
field = ex.attrs['field']

# Finds where to start appending the new files
index=file_names.index(field[-1])

new_files=file_names[index+1:]



if len(new_files)>0:
    for file_name in new_files:
        path=glob.glob(raw + file_name+'*')[0]
        data = to_xarr(path)
        field.append(file_name)
        data.attrs['field'] = field
        data.to_zarr(store, append_dim="time", compute=True)
        print("saved "+file_name,flush=True)
        del data
else:
    print("no files to save",flush=True)
