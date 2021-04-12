import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tkl


def Unet(cascades, 
        shape, 
        filters=24, 
        filters_out=3, 
        pool_size=2, 
        final_activation='relu', 
        batch_normalization=False,
        data_format='channels_last'):
    """ implementation of U-net with defined cascedes number """
    skipcons = []
    bn_axis = -1
    if data_format != 'channels_last':
        bn_axis = 1

    l = [tkl.Input(shape=shape)]
    for c in range(cascades):
        l.append(tkl.Conv2D(filters=filters*2**c, kernel_size=3, padding='same', activation='relu', data_format=data_format)(l[-1]))
        if batch_normalization:
            l.append(tkl.BatchNormalization(axis=bn_axis)(l[-1]))
        l.append(tkl.Conv2D(filters=filters*2**c, kernel_size=3, padding='same', activation='relu', data_format=data_format)(l[-1]))
        skipcons.append(len(l)-1)
        l.append(tkl.MaxPool2D(pool_size=pool_size, strides=pool_size, data_format=data_format)(l[-1]))
        
    l.append(tkl.Conv2D(filters=filters*2**cascades, kernel_size=3, padding='same', activation='relu', data_format=data_format)(l[-1])) 
    l.append(tkl.Conv2D(filters=filters*2**cascades, kernel_size=3, padding='same', activation='relu', data_format=data_format)(l[-1])) 

    for c in range(cascades-1,-1,-1):
        l.append(tkl.Conv2DTranspose(filters=filters*2**c, kernel_size=pool_size, strides=pool_size, activation='relu', data_format=data_format)(l[-1]))
        l.append(tkl.Concatenate()([l[skipcons.pop()], l[-1]]))  
        l.append(tkl.Conv2D(filters=filters*2**c, kernel_size=3, padding='same', activation='relu', data_format=data_format)(l[-1]))  
        if batch_normalization:
            l.append(tkl.BatchNormalization(axis=bn_axis)(l[-1]))
        l.append(tkl.Conv2D(filters=filters*2**c, kernel_size=3, padding='same', activation='relu', data_format=data_format)(l[-1])) 
    
    l.append(tkl.Conv2D(filters=filters_out, kernel_size=3, padding='same', activation=final_activation, data_format=data_format)(l[-1])) 
    
    model = tk.Model(inputs=l[0], outputs=l[-1], name=f"U-net_{cascades}cas{filters}fil{filters_out}out")
    return model