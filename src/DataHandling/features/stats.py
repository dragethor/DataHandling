

def custom_optimize(dsk, keys):
    import dask
    dsk = dask.optimization.inline(dsk, inline_constants=True)
    return dask.array.optimization.optimize(dsk, keys)

def get_valdata(quantity):
    """Returns the data used for validation

    Args:
        quantity ([type]): [description]

    Returns:
        [type]: [description]
    """

    import pandas as pd
    import numpy as np

    if quantity == 'uTexa':
        data = pd.read_csv("/home/au643300/NOBACKUP/data/external/validation/" + quantity + '_valData.csv', sep=' ',
                           skiprows=1, header=None)
        data = data.dropna(axis=1)
    else:
        data = pd.read_csv("/home/au643300/NOBACKUP/data/external/validation/" + quantity + '_valData.csv', sep=' ',
                           skiprows=1, header=None)
        data = data.dropna(axis=1)
        data = data.drop(columns=[0])

    if quantity == 'u':
        data.rename(columns={3: 'y+', 6: 'u_mean', 9: 'uu+', 12: 'ww+'}, inplace=True)
        #Re = 14124
        RMS_adi = np.sqrt(data['uu+'].to_numpy())
        mean_adi = data['u_mean'].to_numpy()

        data[quantity + '_plusmean'] = mean_adi
        data[quantity + '_plusRMS'] = RMS_adi
        data.drop(columns=['u_mean', 'uu+', 'ww+'], inplace=True)


    elif quantity == 'pr1':
        data.rename(columns={3: 'pr1_y+', 6: 'pr1_plusmean', 9: 'pr1_plusRMS', 12: 'ut+'}, inplace=True)
        data.drop(columns=['ut+'], inplace=True)

    elif quantity == 'pr71':
        data.rename(columns={3: 'pr0.71_y+', 6: 'pr0.71_plusmean', 9: 'pr0.71_plusRMS', 12: 'ut+'}, inplace=True)

        data.drop(columns=['ut+'], inplace=True)

    elif quantity == 'pr0025':
        data.rename(columns={3: 'pr0.025_y+', 6: 'pr0.025_plusmean', 9: 'pr0.025_plusRMS', 12: 'ut+'}, inplace=True)

        data = data[['pr0.025_y+', 'pr0.025_plusmean', 'pr0.025_plusRMS']]
    if quantity == 'uTexa':
        data.rename(columns={3: 'y', 6: 'y+', 9: 'u_mean', 12: 'ut+'}, inplace=True)
        RMS_adi = np.zeros(len(data['y']))
        # RMS_adi =  np.sqrt(data['ut+'].to_numpy())
        mean_adi = data['u_mean'].to_numpy()

        data[quantity + '_plusmean'] = mean_adi
        data[quantity + '_plusRMS'] = RMS_adi
        data.drop(columns=['u_mean', 'uu+', 'ww+'], inplace=True)

    # ReTau = 395
    # utau=ReTau/Re

    return data



def calc_stats(ds,save_spot):
    """
    Calculates the time stats and saves them in save_spot. Needs a running dask cluster

    :param ds: dataset
    :param save_spot: where to save the calculated stats
    :return: None
    """
    import xarray as xr
    import dask
    Re_Tau = 395  # Direct from simulation
    Re = 10400  # Direct from simulation
    nu = 1 / Re  # Kinematic viscosity
    u_tau = Re_Tau * nu  # The friction velocity
    val_list=list(ds.keys())
    val_list=sorted(val_list)
    mean=ds
    mean=mean.drop(labels=val_list)
    k=0
    for val in val_list:
        mean_tem=ds[val]/u_tau
        mean_tem=xr.DataArray.mean(mean_tem,dim=('x','z'))
        rms_tem=(((ds[val]/u_tau)-mean_tem)**2)
        rms_tem=xr.ufuncs.sqrt(xr.DataArray.mean(rms_tem,dim=('x','z')))
        mean[val[0]+'_plusmean']=mean_tem
        mean[val[0]+'_plusRMS']=rms_tem
        if k<4: #So for the Pr numbers
            mean=mean.drop(labels=val[0]+'_plusmean')
            mean=mean.drop(labels=val[0]+'_plusRMS')
            pr=float(val[2:])
            theta_tau=mean_tem.differentiate('y')*pr
            theta_plus=mean_tem/theta_tau
            rms_pr=rms_tem/theta_tau
            mean[val+'_plusmean']=theta_plus
            mean[val+'_plusRMS']=rms_pr
        k=k+1

    mean.to_netcdf(save_spot)
