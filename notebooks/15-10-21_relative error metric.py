
#%%
from DataHandling.features import slices
import xarray as xr
from DataHandling import plots
from tensorflow import keras
import os
import numpy as np
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
normalize=True
dropout=False
model_name="sweet-firefly-6"














#%%


data=slices.load_from_scratch(y_plus,var,target,normalized=normalize)

model_path=os.path.join("/home/au643300/DataHandling/models/trained/",model_name)
model=keras.models.load_model(model_path)


predctions=[]
for features in feature_list:
    predctions.append(model.predict(features))

predctions=[np.squeeze(x,axis=3) for x in predctions]



