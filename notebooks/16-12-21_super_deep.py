#%%
import os
from tensorflow import keras
from keras import layers
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from DataHandling.features import slices
from DataHandling import utility
from DataHandling.models import models
os.environ['WANDB_DISABLE_CODE']='True'



# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass



y_plus=15
repeat=3
shuffle=100
batch_size=10
activation='elu'
optimizer="adam"
loss='mean_squared_error'
patience=50
var=['u_vel','v_vel','w_vel']
target=['pr0.71_flux']
normalized=False
dropout=False
skip=4
model_type="super Deep"
data=slices.load_from_scratch(y_plus,var,target,normalized,repeat=repeat,shuffle_size=shuffle,batch_s=batch_size)
train=data[0]
validation=data[1]


model=models.super_deep(var,activation)
model.summary()

#keras.utils.plot_model(model,show_shapes=True,dpi=100)

#%%

wandb.init(project="Thesis",notes="Super deep model")



config=wandb.config
config.y_plus=y_plus
config.repeat=repeat
config.shuffle=shuffle
config.batch_size=batch_size
config.activation=activation
config.optimizer=optimizer
config.loss=loss
config.patience=patience
config.variables=var
config.target=target[0]
config.dropout=dropout
config.normalized=normalized
config.skip=skip
config.model=model_type



model.compile(loss=loss, optimizer=optimizer)


#%%

logdir, backupdir= utility.get_run_dir(wandb.run.name)



backup_cb=tf.keras.callbacks.ModelCheckpoint(os.path.join(backupdir,'weights.{epoch:02d}'),save_best_only=False)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience,
restore_best_weights=True)
model.fit(x=train,epochs=100000,validation_data=validation,callbacks=[WandbCallback(),early_stopping_cb,backup_cb])

model.save(os.path.join("/home/au643300/DataHandling/models/trained",wandb.run.name))