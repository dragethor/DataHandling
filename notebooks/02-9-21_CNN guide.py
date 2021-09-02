#Følger guiden fra 
# https://victorzhou.com/blog/keras-cnn-tutorial/

#%%
import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.python.keras import activations
from tensorflow.keras.utils import to_categorical
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#%%

mnist_dataset = tf.keras.datasets.mnist.load_data(path="mnist.npz")

(x_train, y_train), (x_test, y_test) = mnist_dataset
 


#Normalizing

x_train=(x_train/255)-0.5
x_test=(x_test/255)-0.5

#expanding dimes because of keras

x_train=np.expand_dims(x_train,axis=3)
x_test=np.expand_dims(x_test,axis=3)


#%% making the CNN

num_filters =8
filter_size=3
pool_size=2


model = Sequential([
    Conv2D(num_filters,filter_size,input_shape=(28,28,1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10,activation='softmax')
])

#%%
model.summary()

#%%

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)


#%%

#Ændrer y værdierne til en vektor istedet for et tal

model.fit(
    x_train,
    to_categorical(y_train),
    epochs=3,
    validation_data=(x_test,to_categorical(y_test)),
)


#%%

# Predict on the first 5 test images.
predictions = model.predict(x_test[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(y_test[:5]) # [7, 2, 1, 0, 4]