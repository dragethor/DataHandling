
#%%
from DataHandling.features import slices
import xarray as xr

y_plus=15
repeat=5
shuffle=100
batch_size=10
activation='elu'
optimizer="adam"
loss='mean_squared_error'
patience=100
var=['u_vel']
target=['tau_wall']
normalized=True
dropout=False


df=xr.open_zarr("/home/au643300/DataHandling/data/interim/data.zarr")

#%%

slices.save_tf(y_plus,var,target,df,normalized=True)