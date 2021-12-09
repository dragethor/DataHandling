

#%%

from DataHandling.features import slices
import xarray as xr



df=xr.open_zarr("/home/au643300/DataHandling/data/interim/data.zarr")

#TODO evt kigge på at bruge del til at cleane mellem runs
#TODO hvis det giver mening kan jeg evt tage og returne min lazy array func, og derefter regne dem alle sammen på en gang. Burde gå ekstremt hurtigt. Evt også persiste dem og så trække dem ned når de skal bruges og delte dem bagefter


var=['u_vel',"pr1"]
target=['pr1_flux']
normalized=False
y_plus=15
slices.save_tf(y_plus,var,target,df,normalized=normalized)



var=['u_vel',"pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=15
slices.save_tf(y_plus,var,target,df,normalized=normalized)



var=['u_vel',"pr0.2"]
target=['pr0.2_flux']
normalized=False
y_plus=15
slices.save_tf(y_plus,var,target,df,normalized=normalized)
