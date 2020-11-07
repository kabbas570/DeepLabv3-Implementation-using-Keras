import tensorflow.keras as keras
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
def Deeplab_v3(input_size = (896,896,3),  batchnorm = True):
    inputs = keras.layers.Input(input_size)
    x= Conv2D(3, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(inputs)
    c0 = BatchNormalization()(x)
    c0 = Activation('relu')(c0)
    c0 = Conv2D(32, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c0)
    c0 = Conv2D(32, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c0)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = Conv2D(64, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c0)
    c1 =  Conv2D(64, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)  
    p1 = MaxPooling2D((2, 2))(c1)

    
    c2 = Conv2D(128, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(p1)
    c2 = Conv2D(128, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c2)
    c2 = Conv2D(128, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2) 
    p2 = MaxPooling2D((2, 2))(c2)


    c3 = Conv2D(256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(p2)
    c3 = Conv2D(256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = Conv2D(256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3) 
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(p3)
    c4 = Conv2D(256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c4)
    c4 = Conv2D(256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4) 
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(512, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(p4)
    c5 = Conv2D(512, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c5)
    c5 = Conv2D(512, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5) 
    
    #################################################*****ASPP_v3+******##################################################
    x1 = Conv2D(512, kernel_size = (1, 1), kernel_initializer='he_normal',  padding = 'same')(c5)
    x1 = BatchNormalization()(x1)
    x8 = Conv2D(512, kernel_size = (3, 3), dilation_rate = 6, kernel_initializer='he_normal', padding = 'same')(c5)
    x8 = BatchNormalization()(x8)
    x16 = Conv2D(512, kernel_size = (3, 3), dilation_rate = 12, kernel_initializer='he_normal', padding = 'same')(c5)
    x16 = BatchNormalization()(x16)
    x24 = Conv2D(512, kernel_size = (3, 3), dilation_rate = 18, kernel_initializer='he_normal', padding = 'same')(c5)
    x24 = BatchNormalization()(x24)
    
    img = MaxPooling2D(pool_size=16, strides=16, padding='same')(inputs)
    c = concatenate([x1, x8, x16, x24, img])
    ctr = Conv2D(filters = 256, kernel_size = (1, 1), kernel_initializer = 'he_normal',  padding = 'same')(c)
    ###################################################################################################
    
    # Upsampling
    up = Conv2D(128, (1, 1), kernel_initializer = 'he_normal', activation='relu')(ctr)
    up = UpSampling2D(size=((4,4)), interpolation='bilinear')(up)#x4 times upsample 
    
    up1 = Conv2D(64, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(c3)
    upc = concatenate([up1, up])
    
    up2 = Conv2D(32, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(upc)
    up2 = UpSampling2D(size=((4,4)), interpolation='bilinear')(up2)#x4 times upsample 
    
    outputs = Conv2D(2, (1, 1), activation='sigmoid')(up2)
    model = Model(inputs=inputs, outputs=outputs)
    return model