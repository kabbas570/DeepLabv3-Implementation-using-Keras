import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";
import tensorflow as tf
import tensorflow.keras as    keras
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)



import tensorflow as tf
import tensorflow.keras as    keras
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
import cv2
import glob
import numpy as np
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix

## train data
mask_id = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/XEEEE/unet-master/data/membrane/train/label/*.png')):
    mask_id.append(infile)
image_ = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/XEEEE/unet-master/data/membrane/train/image/*.jpg')):
    image_.append(infile)
    
mask_id_test = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/XEEEE/unet-master/data/membrane/test/label/*.png')):
    mask_id_test.append(infile)
image_tets = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/XEEEE/unet-master/data/membrane/test/img/*.jpg')):
    image_tets.append(infile)    
    
height=256
width=256
def DataGen():  
    img_ = []
    mask_  = []
    for i in range(len(image_)):
        target=np.zeros([256,256,8])
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=image/255
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)       
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)  
        target[:,:,0][np.where(mask==89)]=1
        target[:,:,1][np.where(mask==38)]=1
        target[:,:,2][np.where(mask==14)]=1
        target[:,:,3][np.where(mask==113)]=1
        target[:,:,4][np.where(mask==75)]=1
        target[:,:,5][np.where(mask==128)]=1
        target[:,:,6][np.where(mask==52)]=1
        target[:,:,7][np.where(mask==0)]=1
        #mask = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(target)
       
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_

images,labels = DataGen()
def DataGen_test():  
    img_ = []
    mask_  = []
    
    for i in range(len(image_tets)):
        target=np.zeros([256,256,8])
        image = cv2.imread(image_tets[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=image/255
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)  
        mask = cv2.imread(mask_id_test[i],0)
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)  
        target[:,:,0][np.where(mask==89)]=1
        target[:,:,1][np.where(mask==38)]=1
        target[:,:,2][np.where(mask==14)]=1
        target[:,:,3][np.where(mask==113)]=1
        target[:,:,4][np.where(mask==75)]=1
        target[:,:,5][np.where(mask==128)]=1
        target[:,:,6][np.where(mask==52)]=1
        target[:,:,7][np.where(mask==0)]=1
        #mask = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(target)
       
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_
test_images,test_labels = DataGen_test()
def atrous_spatial_pyramid_pooling(inputs, filters=256, regularizer=None):
    '''
    Atrous Spatial Pyramid Pooling (ASPP) Block
    '''

    pool_height = tf.shape(inputs)[1]
    pool_width = tf.shape(inputs)[2]

    resize_height = pool_height
    resize_width = pool_width

    # Atrous Spatial Pyramid Pooling
    # Atrous 1x1
    aspp1x1 = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizer, name='aspp1x1')(inputs)
    # Atrous 3x3, rate = 6
    aspp3x3_1 =keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=(6, 6), kernel_regularizer=regularizer, name='aspp3x3_1')(inputs)
    # Atrous 3x3, rate = 12
    aspp3x3_2 = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=(12, 12), kernel_regularizer=regularizer, name='aspp3x3_2')(inputs)
    # Atrous 3x3, rate = 18
    aspp3x3_3 = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=(18, 18), kernel_regularizer=regularizer, name='aspp3x3_3')(inputs)

    # Image Level Pooling
    image_feature = tf.reduce_mean(inputs, [1, 2], keepdims=True)
    image_feature = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(image_feature)
    image_feature = tf.image.resize(images=image_feature, size=[resize_height, resize_width], name='image_pool_feature')

    # Merge Poolings
    outputs = tf.concat(values=[aspp1x1, aspp3x3_1, aspp3x3_2, aspp3x3_3, image_feature], axis=3, name='aspp_pools')
    outputs = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizer, name='aspp_outputs')(outputs)
    return outputs
def Xception(
        input_size = (256,256,3)):
    inputs = keras.layers.Input(input_size)
    #32
    x = keras.layers.Conv2D(32, (3, 3),strides=(2, 2), padding='same',name='block1_conv1')(inputs)
    x = keras.layers.BatchNormalization(name='block1_conv1_bn')(x)
    x = keras.layers.Activation('relu', name='block1_conv1_act')(x)
    #64
    x = keras.layers.Conv2D(64, (3, 3), use_bias=False,padding='same', name='block1_conv2')(x)
    x = keras.layers.BatchNormalization( name='block1_conv2_bn')(x)
    x = keras.layers.Activation('relu', name='block1_conv2_act')(x)
    #residual
    residual =  keras.layers.Conv2D(128, (1, 1),strides=(2, 2),padding='same',use_bias=False)(x)
    residual =  keras.layers.BatchNormalization()(residual)
    # 128_2 
    for i in range(2):
        x = keras.layers.SeparableConv2D(128, (3, 3),padding='same',use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
    # 128_ 3x3 stride=2   
    x = keras.layers.SeparableConv2D(128, (3, 3),strides=(2, 2),padding='same',use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.add([x, residual])
    #residual
    residual =  keras.layers.Conv2D(256, (1, 1),strides=(2, 2),padding='same',use_bias=False)(x)
    residual =  keras.layers.BatchNormalization()(residual)
    
     # 256_2 
    for i in range(2):
        x = keras.layers.SeparableConv2D(256, (3, 3),padding='same',use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
    skip1=x
    # 256 3x3 stride=2   
    x = keras.layers.SeparableConv2D(256, (3, 3),strides=(2, 2),padding='same',use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.add([x, residual])
    
     #residual
    residual =  keras.layers.Conv2D(728, (1, 1),strides=(2, 2),padding='same',use_bias=False)(x)
    residual =  keras.layers.BatchNormalization()(residual)
    
     # 256_2 
    for i in range(2):
        x = keras.layers.SeparableConv2D(728, (3, 3),padding='same',use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
    # 728 3x3 stride=2   
    x = keras.layers.SeparableConv2D(728, (3, 3),strides=(2, 2),padding='same',use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.add([x, residual])
    
    for i in range(16):
        residual=x
        x = keras.layers.SeparableConv2D(728, (3, 3),padding='same',use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.SeparableConv2D(728, (3, 3),padding='same',use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.SeparableConv2D(728, (3, 3),padding='same',use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.add([x, residual])       
    #residual
    residual =  keras.layers.Conv2D(1024, (1, 1),strides=(1, 1),padding='same',use_bias=False)(x)
    residual =  keras.layers.BatchNormalization()(residual)
    #728_ 3x3
    x = keras.layers.SeparableConv2D(728, (3, 3),padding='same',use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #1024_ 3x3
    x = keras.layers.SeparableConv2D(1024, (3, 3),padding='same',use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)  
    # 1024 3x3 stride=2
    x = keras.layers.SeparableConv2D(1024, (3, 3),strides=(1, 1),padding='same',use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.add([x, residual])
     #1536_ 3x3
    x = keras.layers.SeparableConv2D(1536, (3, 3),padding='same',use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
     #1536_ 3x3
    x = keras.layers.SeparableConv2D(1536, (3, 3),padding='same',use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
     #2048_ 3x3
    x = keras.layers.SeparableConv2D(2048, (3, 3),padding='same',use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x=atrous_spatial_pyramid_pooling(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x) 
    
    #V3+ Decoder
    x = keras.layers.UpSampling2D(size=((4,4)), interpolation='bilinear')(x)#x4 times upsample
    # 1x1 48
    x_48=  keras.layers.Conv2D(48, 1, padding='same')(skip1)
    x_48 = keras.layers.BatchNormalization()(x_48)
    x_48 = keras.layers.Activation('relu')(x_48)
    # concatinate
    x = keras.layers.Concatenate()([x_48,x])
    # 3x3 256 2 times
    for i in range(2):
        x = keras.layers.Conv2D(256, (3, 3),strides=(1, 1), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
    
    #featuresmaps=number of classes
    x =  keras.layers.Conv2D(8, 1, padding='same')(x)
    #2nd times upsampling
    x = keras.layers.UpSampling2D(size=((4,4)), interpolation='bilinear')(x)#x4 times upsample
    model = keras.models.Model(inputs=inputs, outputs=x, name="DeepLabV3")
    return model
model=Xception()
epochs=100
Adam = optimizers.Adam(lr=0.00001,  beta_1=0.7, beta_2=0.7)
def dice_coef(y_true, y_pred, smooth=2):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
model.summary()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(images,labels,validation_data=(test_images,test_labels),batch_size=2, 
                    epochs=epochs)
model.save_weights("SEG_11_bilinea1r.h5")
model.load_weights("SEG_11_bilinea1r.h5")
result = model.predict(test_images,batch_size=1)
def mean_iou(result,Target):
    P=np.zeros([result.shape[0],256,256,8])
    P[:,:,:,:][np.where(result[:,:,:,:]>0.5)]=1
    predicted=np.reshape(P,(result.shape[0]*8*256*256))
    predicted=predicted.astype(int)
    Target=np.reshape(Target,(result.shape[0]*8*256*256))
    target=Target.astype(int)
    tn, fp, fn, tp=confusion_matrix(target, predicted).ravel()
    print(tp)
    iou=tp/(tp+fn+fp)
    precision=tp/(tp+fp)
    recal=tp/(tp+fn)
    F1=(2*precision*recal)/(precision+recal)
    print("F1_Score is:  ",F1)
    return print("Mean IOU is : " ,iou)
IOU_result=mean_iou(result,test_labels)

P=np.zeros([12,256,256,8])
P[:,:,:,:][np.where(result[:,:,:,:]>0.5)]=1
import os
P3='/home/user01/data_ssd/Abbas/XEEEE/unet-master/data/membrane/results/'
for i in range(len(result)):
    img=np.zeros([256,256,3]) 
    l1=P[i,:,:,:]
    k1=np.argmax(l1, axis=-1)
    img[np.where(k1==0)]=[128,128,0]
    img[np.where(k1==1)]=[0,0,128]
    img[np.where(k1==2)]=[128,0,0]
    img[np.where(k1==3)]=[0,128,0]
    img[np.where(k1==4)]=[0,128,0]
    img[np.where(k1==5)]=[128,128,128]
    img[np.where(k1==6)]=[128,0,128]
    img[np.where(k1==7)]=[0,0,0]
    cv2.imwrite(os.path.join(P3 , str(i)+".png"),img) 