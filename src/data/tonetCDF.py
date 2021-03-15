


# TODO Det er ret vigtigt at der kun er en tråd pr worker når det her kører





def to_netcdf(file_path):
    """
    :param file: the filename of the file there is to be converted to netCDF
    :return:
    """ 
    from src.data.read_binary import readDNSdata
    import xarray as xr
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

    )
    ds=ds.chunk(1000)
    # Saving the files as netcdf files
    # print('saved'+file_path[-12:-1])
    # ds.to_netcdf(save_path + file_path[-12:-1] + 'nc', engine='netcdf4')

    return ds

