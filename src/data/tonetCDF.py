

from read_binary import *
import os
from dask.distributed import Client, as_completed, LocalCluster, get_task_stream
import xarray as xr
from dask_jobqueue import SLURMCluster
import xarray as xr
from dask.distributed import Client, fire_and_forget
import glob



# TODO Det er ret vigtigt at der kun er en tråd pr worker når det her kører

def to_netcdf(file_path):
    """
    :param file: the filename of the file there is to be converted to netCDF
    :return:
    """
    save_path = "/home/au643300/NOBACKUP/interim/"


    # Sorting the list of files in the raw_path dir


    # Here a complete list of the raw file path incl the file name is made


    quantities, _, xF, yF, zF, length, _, _ = readDNSdata(file_path)
    # As the files are completed they are saved as a xarray dataset

    xF = xF[:-1]
    zF = zF[:-1]
    ds = xr.Dataset(
        {
            "u_vel": (['x', 'y', 'z'], quantities[0]),
            "v_vel": (['x', 'y', 'z'], quantities[1]),
            "w_vel": (['x', 'y', 'z'], quantities[2]),
            "pr1": (['x', 'y', 'z'], quantities[3]),
            "pr0.71": (['x', 'y', 'z'], quantities[4]),
            "pr0.2": (['x', 'y', 'z'], quantities[5]),
            "pr0.025": (['x', 'y', 'z'], quantities[6]),
        },
        coords={
            "x": (["x"], xF),
            "y": (["y"], yF),
            "z": (["z"], zF),
            "time": length[2],
        },

    ).chunk(1000000)
    # Saving the files as netcdf files
    print('saved'+file_path[-12:-1])
    ds.to_netcdf(save_path + file_path[-12:-1] + 'nc')



#memory and cores are pr JOB and not in total
#Setup for q64 only so far
cluster=SLURMCluster(cores=2,
                     processes=1,
                     memory="11GB",
                     queue='q64',
                     walltime='0-01:00:00',
                     local_directory='/scratch/$SLURM_JOB_ID',
                     interface='ib0',
                     )




cluster.scale(20)


client =Client(cluster)
print(client.dashboard_link)



raw_path="/home/au643300/DataHandling/data/raw/"

filelist=glob.glob(raw_path+'*.u')
file_path=[]

for path in filelist:
    file_path.append(os.readlink(path))



a=client.map(to_netcdf,filelist)

fire_and_forget(a)


