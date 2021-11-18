

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
shuffle=200
batch_size=10
activation='elu'
optimizer="adam"
loss='mean_squared_error'
patience=50
var=['u_vel']
target=['pr0.71_flux']
normalized=False
dropout=False
skip=3
data=slices.load_from_scratch(y_plus,var,target,normalized,repeat=repeat,shuffle_size=shuffle,batch_s=batch_size)
train=data[0]
validation=data[1]


#%%


# #Model
# input_img = layers.Input(shape=(256,256),name=var[0])
# input_reshape=layers.Reshape((256,256,1))(input_img)
# #Down sampled skip-connection model
# down_1 = layers.MaxPooling2D((8,8),padding='same')(input_reshape)
# x1 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(down_1)
# x1 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x1)
# x1 = layers.UpSampling2D((2,2))(x1)

# down_2 = layers.MaxPooling2D((4,4),padding='same')(input_reshape)
# x2 = layers.Concatenate()([x1,down_2])
# x2 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x2)
# x2 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x2)
# x2 = layers.UpSampling2D((2,2))(x2)

# down_3 = layers.MaxPooling2D((2,2),padding='same')(input_reshape)
# x3 = layers.Concatenate()([x2,down_3])
# x3 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x3)
# x3 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x3)
# x3 = layers.UpSampling2D((2,2))(x3)

# x4 = layers.Concatenate()([x3,input_reshape])
# x4 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x4)
# x4 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x4)

# #Multi-scale model (Du et al., 2018)
# layer_1 = layers.Conv2D(16, (5,5),activation=activation, padding='same')(input_reshape)
# x1m = layers.Conv2D(8, (5,5),activation=activation, padding='same')(layer_1)
# x1m = layers.Conv2D(8, (5,5),activation=activation, padding='same')(x1m)

# layer_2 = layers.Conv2D(16, (9,9),activation=activation, padding='same')(input_reshape)
# x2m = layers.Conv2D(8, (9,9),activation=activation, padding='same')(layer_2)
# x2m = layers.Conv2D(8, (9,9),activation=activation, padding='same')(x2m)

# layer_3 = layers.Conv2D(16, (13,13),activation=activation, padding='same')(input_reshape)
# x3m = layers.Conv2D(8, (13,13),activation=activation, padding='same')(layer_3)
# x3m = layers.Conv2D(8, (13,13),activation=activation, padding='same')(x3m)

# x_add = layers.Concatenate()([x1m,x2m,x3m,input_reshape])
# x4m = layers.Conv2D(8, (7,7),activation=activation,padding='same')(x_add)
# x4m = layers.Conv2D(3, (5,5),activation=activation,padding='same')(x4m)

# x_final = layers.Concatenate()([x4,x4m])
# x_final = layers.Conv2D(1, (3,3),padding='same')(x_final)
# model = keras.Model(input_img, x_final)



#keras.utils.plot_model(model,show_shapes=True,dpi=100)


#%%

model=keras.models.load_model('/home/au643300/DataHandling/models/backup/run_2021_11_17-06_03-icy-fire-52')

#%%
#Wandb stuff
wandb_id="9paou0fo"
wandb.init(project="Thesis",notes="Heat with fukiama superres model",id=wandb_id,resume="auto")


# config=wandb.config
# config.y_plus=y_plus
# config.repeat=repeat
# config.shuffle=shuffle
# config.batch_size=batch_size
# config.activation=activation
# config.optimizer=optimizer
# config.loss=loss
# config.patience=patience
# config.variables=var
# config.target=target[0]
# config.dropout=dropout
# config.normalized=normalized
# config.skip=skip



#model.compile(loss=loss, optimizer=optimizer)


#%%

logdir, backupdir= utility.get_run_dir(wandb.run.name)



backup_cb=tf.keras.callbacks.ModelCheckpoint(backupdir,save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience,
restore_best_weights=True)
model.fit(x=train,epochs=100000,validation_data=validation,callbacks=[WandbCallback(),early_stopping_cb,backup_cb])

model.save(os.path.join("/home/au643300/DataHandling/models/trained",wandb.run.name))

