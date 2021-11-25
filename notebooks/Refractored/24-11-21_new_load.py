

#%%

from DataHandling.features import slices
import importlib
from DataHandling import utility
importlib.reload(slices)
importlib.reload(utility)
import tensorflow as tf
y_plus=15
activation='elu'
optimizer="adam"
loss='mean_squared_error'
var=['u_vel','v_vel','w_vel']
target=['pr0.71_flux']
target_type='flux'
normalize=False
model_name='playful-night-58'

#%%
# a=slices.load_validation(y_plus,var,target,normalize)


# train=a[0]
# val=a[1]
# test=a[2]

#%%
feature_list,target_list,predctions,names=utility.get_data(model_name,y_plus,var,target,normalize)
