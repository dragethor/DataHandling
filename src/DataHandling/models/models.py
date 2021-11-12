



def baseline_cnn(input_feature,activation='elu'):
    
    activation='elu'
    from tensorflow import keras
    import tensorflow as tf

    weights=[128,256,256]
    input=keras.layers.Input(shape=(256,256),name=input_feature[0])
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

    model = keras.Model(inputs=input, outputs=output, name="Baseline")
    return model


def baseline_cnn_dropout(input_feature,activation='elu'):
    
    
    from tensorflow import keras
    import tensorflow as tf

    weights=[128,256,256]
    input=keras.layers.Input(shape=(256,256),name=input_feature[0])
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

    model = keras.Model(inputs=input, outputs=output, name="dropout")
    return model




def baseline_cnn_skip1(input_feature,activation='elu'):
    
    
    from tensorflow import keras
    import tensorflow as tf

    weights=[128,256,256]
    input=keras.layers.Input(shape=(256,256),name=input_feature[0])
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

    model = keras.Model(inputs=input, outputs=output, name="1_skip_conn")
    return model


def baseline_cnn_multipel_inputs(input_features,activation='elu'):
    
    
    from tensorflow import keras
    import tensorflow as tf
    input_list=[]
    reshape_list=[]

    weights=[128,256,256]
    
    for features in input_features:
        input=keras.layers.Input(shape=(256,256),name=features)
        input_list.append(input)
        reshape=keras.layers.Reshape((256,256,1))(input)
        reshape_list.append(reshape)
    conc=keras.layers.Concatenate()(reshape_list)

    batch=keras.layers.BatchNormalization(-1)(conc)
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

    model = keras.Model(inputs=input_list, outputs=output, name="Multi input")
    return model

def baseline_cnn_sep(input_feature,activation='elu'):
    
    activation='elu'
    from tensorflow import keras
    import tensorflow as tf

    weights=[128,256,256]
    input=keras.layers.Input(shape=(256,256),name=input_feature[0])
    reshape=keras.layers.Reshape((256,256,1))(input)
    batch=keras.layers.BatchNormalization(-1)(reshape)
    cnn=keras.layers.Conv2D(64,5,activation=activation)(batch)
    batch=keras.layers.BatchNormalization(-1)(cnn)
    for weight in weights:
        cnn=keras.layers.SeparableConv2D(weight,3,activation=activation)(batch)
        batch=keras.layers.BatchNormalization(-1)(cnn)
        
    for weight in reversed(weights):
        cnn=keras.layers.Conv2DTranspose(weight,3,activation=activation)(batch)
        batch=keras.layers.BatchNormalization(-1)(cnn)



    cnn=tf.keras.layers.Conv2DTranspose(64,5,activation=activation)(batch)
    batch=keras.layers.BatchNormalization(-1)(cnn)
    output=tf.keras.layers.Conv2DTranspose(1,1)(batch)

    model = keras.Model(inputs=input, outputs=output, name="SepConv2d")
    return model


def baseline_cnn_no_BN(input_feature,activation='elu'):
    
    activation='elu'
    from tensorflow import keras
    import tensorflow as tf

    weights=[128,256,256]
    input=keras.layers.Input(shape=(256,256),name=input_feature[0])
    reshape=keras.layers.Reshape((256,256,1))(input)
    cnn=keras.layers.Conv2D(64,5,activation=activation)(reshape)
    for weight in weights:
        cnn=keras.layers.Conv2D(weight,3,activation=activation)(cnn)
        
    for weight in reversed(weights):
        cnn=keras.layers.Conv2DTranspose(weight,3,activation=activation)(cnn)



    cnn=tf.keras.layers.Conv2DTranspose(64,5,activation=activation)(cnn)
    output=tf.keras.layers.Conv2DTranspose(1,1)(cnn)

    model = keras.Model(inputs=input, outputs=output, name="No_BN")
    return model