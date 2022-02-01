

#%%


""" Bruges til at lave nye slices af dataen """



from DataHandling.features import slices
import xarray as xr



df=xr.open_zarr("/home/au643300/DataHandling/data/interim/data.zarr")



var=['u_vel']
target=['tau_wall']
normalized=False
y_plus=15
slices.save_tf(y_plus,var,target,df,normalized=normalized)

