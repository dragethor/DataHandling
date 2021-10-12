



def baseline_cnn(activation='elu'):
    
    
    from tensorflow import keras
    import tensorflow as tf

    weights=[128,256,256]
    input=keras.layers.Input(shape=(256,256),name='u_vel')
    reshape=keras.layers.Reshape((256,256,1))(input)
    batch=keras.layers.BatchNormalization(-1)(reshape)
    cnn=keras.layers.Conv2D(64,5,activation=activation)(batch)
    batch=keras.layers.BatchNormalization(-1)(cnn)
    for weight in weights:
        cnn=keras.layers.Conv2D(weight,3,activation=activation)(batch)
        batch=keras.layers.BatchNormalization(-1)(cnn)
        
    for weight in reversed(weights):
        cnn=keras.layers.Conv2DTranspose(weight,3,activation=activation)(batch)
        batch=keras.layers.BatchNormalization(-1)(cnn)



    cnn=tf.keras.layers.Conv2DTranspose(64,5,activation=activation)(batch)
    batch=keras.layers.BatchNormalization(-1)(cnn)
    output=tf.keras.layers.Conv2DTranspose(1,1)(batch)

    model = keras.Model(inputs=input, outputs=output, name="CNN_baseline")
    return model


def baseline_cnn_dropout(activation='elu'):
    
    
    from tensorflow import keras
    import tensorflow as tf

    weights=[128,256,256]
    input=keras.layers.Input(shape=(256,256),name='u_vel')
    reshape=keras.layers.Reshape((256,256,1))(input)
    batch=keras.layers.BatchNormalization(-1)(reshape)
    drop=keras.layers.Dropout(0.4)(batch)
    cnn=keras.layers.Conv2D(64,5,activation=activation)(drop)
    batch=keras.layers.BatchNormalization(-1)(cnn)
    drop=keras.layers.Dropout(0.4)(batch)
    for weight in weights:
        cnn=keras.layers.Conv2D(weight,3,activation=activation)(drop)
        batch=keras.layers.BatchNormalization(-1)(cnn)
        drop=keras.layers.Dropout(0.4)(batch)
        
    for weight in reversed(weights):
        cnn=keras.layers.Conv2DTranspose(weight,3,activation=activation)(batch)
        batch=keras.layers.BatchNormalization(-1)(cnn)



    cnn=tf.keras.layers.Conv2DTranspose(64,5,activation=activation)(batch)
    batch=keras.layers.BatchNormalization(-1)(cnn)
    output=tf.keras.layers.Conv2DTranspose(1,1)(batch)

    model = keras.Model(inputs=input, outputs=output, name="CNN_baseline")
    return model




def baseline_cnn_skip1(activation='elu'):
    
    
    from tensorflow import keras
    import tensorflow as tf

    weights=[128,256,256]
    input=keras.layers.Input(shape=(256,256),name='u_vel')
    reshape=keras.layers.Reshape((256,256,1))(input)
    batch=keras.layers.BatchNormalization(-1)(reshape)
    cnn=keras.layers.Conv2D(64,5,activation=activation)(batch)
    batch=keras.layers.BatchNormalization(-1)(cnn)
    for weight in weights:
        cnn=keras.layers.Conv2D(weight,3,activation=activation)(batch)
        batch=keras.layers.BatchNormalization(-1)(cnn)
        
    for weight in reversed(weights):
        cnn=keras.layers.Conv2DTranspose(weight,3,activation=activation)(batch)
        batch=keras.layers.BatchNormalization(-1)(cnn)



    cnn=tf.keras.layers.Conv2DTranspose(64,5,activation=activation)(batch)
    batch=keras.layers.BatchNormalization(-1)(cnn)
    conc=keras.layers.Concatenate()([batch,reshape])
    output=tf.keras.layers.Conv2DTranspose(1,1)(conc)

    model = keras.Model(inputs=input, outputs=output, name="CNN_baseline")
    return model