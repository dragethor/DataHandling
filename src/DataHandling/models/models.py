




def baseline_cnn(input_feature,activation='elu'):
    
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

    model = keras.Model(inputs=input_list, outputs=output)
    return model

def baseline_cnn_sep_skip_final(input_features,activation='elu'):
    
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
    

    batch1=keras.layers.BatchNormalization(-1)(conc)
    cnn1=keras.layers.Conv2D(64,5,activation=activation)(batch1)
    
    batch2=keras.layers.BatchNormalization(-1)(cnn1)
    cnn2=keras.layers.SeparableConv2D(weights[0],3,activation=activation)(batch2)
    batch3=keras.layers.BatchNormalization(-1)(cnn2)
    cnn3=keras.layers.SeparableConv2D(weights[1],3,activation=activation)(batch3)
    batch4=keras.layers.BatchNormalization(-1)(cnn3)
    cnn4=keras.layers.SeparableConv2D(weights[2],3,activation=activation)(batch4)
    batch5=keras.layers.BatchNormalization(-1)(cnn4)

    conc1=keras.layers.Concatenate()([cnn4,batch5])
    cnn5=keras.layers.Conv2DTranspose(weights[0],3,activation=activation)(conc1)
    batch6=keras.layers.BatchNormalization(-1)(cnn5)
    
    conc2=keras.layers.Concatenate()([cnn3,batch6])
    cnn6=keras.layers.Conv2DTranspose(weights[1],3,activation=activation)(conc2)
    batch7=keras.layers.BatchNormalization(-1)(cnn6)
    
    conc3=keras.layers.Concatenate()([cnn2,batch7])
    cnn7=keras.layers.Conv2DTranspose(weights[2],3,activation=activation)(conc3)
    batch8=keras.layers.BatchNormalization(-1)(cnn7)
    
    conc4=keras.layers.Concatenate()([cnn1,batch8])
    cnn8=tf.keras.layers.Conv2DTranspose(64,5,activation=activation)(conc4)
    batch9=keras.layers.BatchNormalization(-1)(cnn8)

    conc5=keras.layers.Concatenate()([conc,batch9])
    output=tf.keras.layers.Conv2DTranspose(1,1)(conc5)

    model = keras.Model(inputs=input_list, outputs=output)
    return model


def baseline_cnn_no_BN(input_feature,activation='elu'):
    
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


def fukiama_model(input_features,activation="elu"):
    
    from tensorflow import keras
    import tensorflow as tf
    from keras import layers
    
    input_list=[]
    reshape_list=[]

    for features in input_features:
        input=keras.layers.Input(shape=(256,256),name=features)
        input_list.append(input)
        reshape=keras.layers.Reshape((256,256,1))(input)
        reshape_list.append(reshape)
    
    conc=keras.layers.Concatenate()(reshape_list)

    #Down sampled skip-connection model
    down_1 = layers.MaxPooling2D((8,8),padding='same')(conc)
    x1 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(down_1)
    x1 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x1)
    x1 = layers.UpSampling2D((2,2))(x1)

    down_2 = layers.MaxPooling2D((4,4),padding='same')(conc)
    x2 = layers.Concatenate()([x1,down_2])
    x2 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x2)
    x2 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x2)
    x2 = layers.UpSampling2D((2,2))(x2)

    down_3 = layers.MaxPooling2D((2,2),padding='same')(conc)
    x3 = layers.Concatenate()([x2,down_3])
    x3 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x3)
    x3 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x3)
    x3 = layers.UpSampling2D((2,2))(x3)

    x4 = layers.Concatenate()([x3,conc])
    x4 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x4)
    x4 = layers.Conv2D(32, (3,3),activation=activation, padding='same')(x4)

    #Multi-scale model (Du et al., 2018)
    layer_1 = layers.Conv2D(16, (5,5),activation=activation, padding='same')(conc)
    x1m = layers.Conv2D(8, (5,5),activation=activation, padding='same')(layer_1)
    x1m = layers.Conv2D(8, (5,5),activation=activation, padding='same')(x1m)

    layer_2 = layers.Conv2D(16, (9,9),activation=activation, padding='same')(conc)
    x2m = layers.Conv2D(8, (9,9),activation=activation, padding='same')(layer_2)
    x2m = layers.Conv2D(8, (9,9),activation=activation, padding='same')(x2m)

    layer_3 = layers.Conv2D(16, (13,13),activation=activation, padding='same')(conc)
    x3m = layers.Conv2D(8, (13,13),activation=activation, padding='same')(layer_3)
    x3m = layers.Conv2D(8, (13,13),activation=activation, padding='same')(x3m)

    x_add = layers.Concatenate()([x1m,x2m,x3m,conc])
    x4m = layers.Conv2D(8, (7,7),activation=activation,padding='same')(x_add)
    x4m = layers.Conv2D(3, (5,5),activation=activation,padding='same')(x4m)

    x_final = layers.Concatenate()([x4,x4m])
    x_final = layers.Conv2D(1, (3,3),padding='same')(x_final)
    model = keras.Model(input_list, x_final)  

    return model  
        


def fukiama_model_more_filters(input_features,activation="elu"):
    
    from tensorflow import keras
    import tensorflow as tf
    from keras import layers
    
    input_list=[]
    reshape_list=[]

    for features in input_features:
        input=keras.layers.Input(shape=(256,256),name=features)
        input_list.append(input)
        reshape=keras.layers.Reshape((256,256,1))(input)
        reshape_list.append(reshape)
    
    conc=keras.layers.Concatenate()(reshape_list)

    #Down sampled skip-connection model
    down_1 = layers.MaxPooling2D((8,8),padding='same')(conc)
    x1 = layers.Conv2D(64, (3,3),activation=activation, padding='same')(down_1)
    x1 = layers.Conv2D(64, (3,3),activation=activation, padding='same')(x1)
    x1 = layers.UpSampling2D((2,2))(x1)

    down_2 = layers.MaxPooling2D((4,4),padding='same')(conc)
    x2 = layers.Concatenate()([x1,down_2])
    x2 = layers.Conv2D(64, (3,3),activation=activation, padding='same')(x2)
    x2 = layers.Conv2D(64, (3,3),activation=activation, padding='same')(x2)
    x2 = layers.UpSampling2D((2,2))(x2)

    down_3 = layers.MaxPooling2D((2,2),padding='same')(conc)
    x3 = layers.Concatenate()([x2,down_3])
    x3 = layers.Conv2D(64, (3,3),activation=activation, padding='same')(x3)
    x3 = layers.Conv2D(64, (3,3),activation=activation, padding='same')(x3)
    x3 = layers.UpSampling2D((2,2))(x3)

    x4 = layers.Concatenate()([x3,conc])
    x4 = layers.Conv2D(64, (3,3),activation=activation, padding='same')(x4)
    x4 = layers.Conv2D(64, (3,3),activation=activation, padding='same')(x4)

    #Multi-scale model (Du et al., 2018)
    layer_1 = layers.Conv2D(64, (5,5),activation=activation, padding='same')(conc)
    x1m = layers.Conv2D(32, (5,5),activation=activation, padding='same')(layer_1)
    x1m = layers.Conv2D(32, (5,5),activation=activation, padding='same')(x1m)

    layer_2 = layers.Conv2D(64, (9,9),activation=activation, padding='same')(conc)
    x2m = layers.Conv2D(32, (9,9),activation=activation, padding='same')(layer_2)
    x2m = layers.Conv2D(32, (9,9),activation=activation, padding='same')(x2m)

    layer_3 = layers.Conv2D(64, (13,13),activation=activation, padding='same')(conc)
    x3m = layers.Conv2D(32, (13,13),activation=activation, padding='same')(layer_3)
    x3m = layers.Conv2D(32, (13,13),activation=activation, padding='same')(x3m)

    x_add = layers.Concatenate()([x1m,x2m,x3m,conc])
    x4m = layers.Conv2D(32, (7,7),activation=activation,padding='same')(x_add)
    x4m = layers.Conv2D(12, (5,5),activation=activation,padding='same')(x4m)

    x_final = layers.Concatenate()([x4,x4m])
    x_final = layers.Conv2D(1, (3,3),padding='same')(x_final)
    model = keras.Model(input_list, x_final)  

    return model  


def final_skip_no_sep(input_features,activation='elu'):
    
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
    

    batch1=keras.layers.BatchNormalization(-1)(conc)
    cnn1=keras.layers.Conv2D(64,5,activation=activation)(batch1)
    
    batch2=keras.layers.BatchNormalization(-1)(cnn1)
    cnn2=keras.layers.Conv2D(weights[0],3,activation=activation)(batch2)
    batch3=keras.layers.BatchNormalization(-1)(cnn2)
    cnn3=keras.layers.Conv2D(weights[1],3,activation=activation)(batch3)
    batch4=keras.layers.BatchNormalization(-1)(cnn3)
    cnn4=keras.layers.Conv2D(weights[2],3,activation=activation)(batch4)
    batch5=keras.layers.BatchNormalization(-1)(cnn4)

    conc1=keras.layers.Concatenate()([cnn4,batch5])
    cnn5=keras.layers.Conv2DTranspose(weights[0],3,activation=activation)(conc1)
    batch6=keras.layers.BatchNormalization(-1)(cnn5)
    
    conc2=keras.layers.Concatenate()([cnn3,batch6])
    cnn6=keras.layers.Conv2DTranspose(weights[1],3,activation=activation)(conc2)
    batch7=keras.layers.BatchNormalization(-1)(cnn6)
    
    conc3=keras.layers.Concatenate()([cnn2,batch7])
    cnn7=keras.layers.Conv2DTranspose(weights[2],3,activation=activation)(conc3)
    batch8=keras.layers.BatchNormalization(-1)(cnn7)
    
    conc4=keras.layers.Concatenate()([cnn1,batch8])
    cnn8=tf.keras.layers.Conv2DTranspose(64,5,activation=activation)(conc4)
    batch9=keras.layers.BatchNormalization(-1)(cnn8)

    conc5=keras.layers.Concatenate()([conc,batch9])
    output=tf.keras.layers.Conv2DTranspose(1,1)(conc5)

    model = keras.Model(inputs=input_list, outputs=output)
    return model




def super_deep(input_features,activation='elu'):
    
    
    from tensorflow import keras
    import tensorflow as tf
    input_list=[]
    reshape_list=[]

    weights=[128,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]
    
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

    model = keras.Model(inputs=input_list, outputs=output)
    return model