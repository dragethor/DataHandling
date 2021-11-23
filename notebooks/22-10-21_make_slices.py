

#%%

from DataHandling.features import slices
import xarray as xr


var=['u_vel','v_vel','w_vel']
var1=['u_vel','v_vel','w_vel','pr0.71']
target=['pr0.71_flux']
normalized=False
y_plus=15

df=xr.open_zarr("/home/au643300/DataHandling/data/interim/data.zarr")


#%%


slices.save_tf(y_plus,var,target,df,normalized=normalized)
#slices.save_tf(y_plus,var1,target,df,normalized=normalized)
