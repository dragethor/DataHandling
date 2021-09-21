
#%%
def to_xarr(file_path):
    """
    takes a raw *.u file and loads its into xarr
    :param file_path: the filename of the file there is to be converted to netCDF
    :return:
    """

    import xarray as xr
    from DataHandling.data_raw.make_dataset import readDNSdata
    # Sorting the list of files in the raw_path dir

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






store="/home/au643300/DataHandling/data/interim/data.zarr"
"""
appends or makes a completly new zarr with all timesteps found in the folder raw.
:param store: Location to store the zarr folder:
:return: nothing. only saves the file
"""





