
#%%
def to_xarr(file_path):
    """
    takes a raw *.u file and loads its into xarr
    :param file_path: the filename of the file there is to be converted to netCDF
    :return:
    """

    import xarray as xr
    from DataHandling.data_raw.make_dataset import readDNSdata

    # Here a complete list of the raw file path incl the file name is made

    quantities, _, xf, yf, zf, length, _, _ = readDNSdata(file_path)
    # As the files are completed they are saved as a xarray dataset

    xf = xf[:-1]
    zf = zf[:-1]
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
            "x": (["x"], xf),
            "y": (["y"], yf),
            "z": (["z"], zf),
            "time": length[2],
        },

    )
    # Saving the files as netcdf files
    # print('saved'+file_path[-12:-1])
    # ds.to_netcdf(save_path + file_path[-12:-1] + 'nc', engine='netcdf4')
    ds = ds.expand_dims("time")
    ds = ds.chunk(10000)
    return ds




def netcdf_save(interim="/home/au643300/NOBACKUP/data/interim/snapshots/"):
    """Saves .u files to netcdf files

    Args:
        interim (str, optional): the default save location. Defaults to "/home/au643300/NOBACKUP/data/interim/snapshots/".
    """

    import glob
    from hashlib import new
    import os
    import numpy as np
    import xarray as xr

    raw = "/home/au643300/DataHandling/data/raw/"

    #%%
    raw_files = glob.glob(raw + '*.u')
    raw_files = sorted(raw_files)
    raw_files_names = [os.path.basename(path) for path in raw_files]
    raw_files_names = [file[0:-2] for file in raw_files_names]

    #raw_files_index=[file[6:] for file in raw_files_names]
    #raw_files_index=np.array(raw_files_index)
    #%%



    #files in interim snapshots folder

    inter_files = glob.glob(interim + '*.nc')
    inter_files = sorted(inter_files)
    inter_files_names = [os.path.basename(path) for path in inter_files]
    inter_files_names = [file[0:-3] for file in inter_files_names]

    #now remove all that are allready in the save folder

    new_files=[x for x in raw_files_names if x not in inter_files_names]


    for name in new_files:
        data=to_xarr(raw+name+".u")
        data.to_netcdf(interim + name + '.nc', engine='netcdf4')
        del data


return None

