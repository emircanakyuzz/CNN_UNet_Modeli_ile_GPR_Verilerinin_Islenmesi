import os
import sys
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn import preprocessing
from sklearn.preprocessing import scale
import tensorflow as tf
from tensorflow import keras

N_train = [2500,2500,2500,2500,2500,2500]
N_test_start = [2500,2500,2500,2500,2500,2500]
N_test = [250,250,250,250,250,250]
N_class = len(N_train)
# Veri boyutlarımızın nasıl olması gerektiğini tanımlıyoruz.
data_sizeX = 200
data_sizeY = 600
mask_sizeX = 256
mask_sizeY = 256
num_channel = 1
# Verileri içerisine aktaracağımız dizi türündeki değişkenleri tanımlıyoruz.
train_data1 = []
train_data2 = []
train_mask= []
test_data1 = []
test_data2 = []
test_mask = []
path = '' # Buraya verilerinizin bilgisayardaki dosya konumu gelmeli 
sub_path = ''

# Verileri bilgisayardan çalışmamıza yükleme işlemini gerçekleştiriyoruz.
for i in range(N_class):
    for j in range(N_train[i]):
        data1 = sio.loadmat(path+'dataset/%d/data1/%d.mat'%(i+1, j+1))['data1']
        train_data1.append(data1.reshape((data_sizeX,data_sizeY,1)))
        data2 = sio.loadmat(path+'dataset/%d/data2/%d.mat'%(i+1, j+1))['data2']
        train_data2.append(data2.reshape((data_sizeX,data_sizeY,1)))
        mask = sio.loadmat(path+'dataset/%d/mask/%d.mat'%(i+1, j+1))['mask']
        train_mask.append(mask.reshape((mask_sizeX,mask_sizeY,1)))
    
    for j in range(N_test_start[i], N_test_start[i]+N_test[i]):
        data1 = sio.loadmat(path+'dataset/%d/data1/%d.mat'%(i+1, j+1))['data1']
        test_data1.append(data1.reshape((data_sizeX,data_sizeY,1)))
        data2 = sio.loadmat(path+'dataset/%d/data2/%d.mat'%(i+1, j+1))['data2']
        test_data2.append(data2.reshape((data_sizeX,data_sizeY,1)))
        mask = sio.loadmat(path+'dataset/%d/mask/%d.mat'%(i+1, j+1))['mask']
        test_mask.append(mask.reshape((mask_sizeX,mask_sizeY,1)))

# Verileri dizi türündeki değişkenlere aktarıyoruz. Numpy dizileri verilerde daha kolaylıkla işlem yapmamıza olanak sağlamaktadır.
train_data1 = np.array(train_data1)
train_data2 = np.array(train_data2)
train_mask = np.array(train_mask)
test_data1 = np.array(test_data1)
test_data2 = np.array(test_data2)
test_mask = np.array(test_mask)

print("veriler basariyla okundu...")

def create_unet_model():
    f0 = 16
    f = [f0, f0*2, f0*4, f0*8, f0*16, f0*32]

    inputs1 = keras.layers.Input((data_sizeX, data_sizeY, num_channel))
    inputs2 = keras.layers.Input((data_sizeX, data_sizeY, num_channel))

    
    p10 = inputs1
    p11 = keras.layers.Conv2D(f[0], (3, 3), padding='same', activation="relu", strides=1)(p10) 
    p11 = keras.layers.Conv2D(f[0], (3,3), padding='same', strides=1,activation="relu")(p11)
    p11=keras.layers.MaxPool2D((2, 2), (2, 2))(p11)
    
    p12 = keras.layers.Conv2D(f[1], (3, 3), padding='same', activation="relu", strides=1)(p11)
    p12 = keras.layers.Conv2D(f[1], (3,3), padding='same', strides=1,activation="relu")(p12)
    p12=keras.layers.MaxPool2D((2, 2), (2, 2))(p12)
    
    p13 = keras.layers.Conv2D(f[2], (3, 3), padding='same', activation="relu", strides=1)(p12)
    p13 = keras.layers.Conv2D(f[2], (3,3), padding='same', strides=1)(p13)
    p13=keras.layers.MaxPool2D((2, 2), (2, 2))(p13)
    
    p14 = keras.layers.Conv2D(f[3], (3, 3), padding='same', activation="relu", strides=1)(p13)
    p14 = keras.layers.Conv2D(f[3], (3,3), padding='same', strides=1)(p14)
    p14=keras.layers.MaxPool2D((2, 2), (2, 2))(p14)
    
    p15 =keras.layers.Conv2D(f[4],(3,3),padding="same",activation="relu",strides=1)(p14)
    p15 =keras.layers.Conv2D(f[4],(3,3),padding="same",activation="relu",strides=1)(p15)
    p15= keras.layers.MaxPool2D((2, 2), (2, 2))(p15)  

    
    

    p20 = inputs2
    p21 = keras.layers.Conv2D(f[0], (3, 3), padding='same', activation="relu", strides=1)(p20)
    p21 = keras.layers.Conv2D(f[0], (3,3), padding='same', strides=1,activation="relu")(p21)
    p21 = keras.layers.MaxPool2D((2, 2), (2, 2))(p21)
    
    
    p22 = keras.layers.Conv2D(f[1], (3, 3), padding='same', activation="relu", strides=1)(p21)
    p22 = keras.layers.Conv2D(f[1], (3,3), padding='same', strides=1,activation="relu")(p22)
    p22 = keras.layers.MaxPool2D((2, 2), (2, 2))(p22)
    
    p23 = keras.layers.Conv2D(f[2], (3, 3), padding='same', activation="relu", strides=1)(p22)
    p23 = keras.layers.Conv2D(f[2], (3,3), padding='same', strides=1)(p23)
    p23=keras.layers.MaxPool2D((2, 2), (2, 2))(p23)
    
    p24 = keras.layers.Conv2D(f[3], (3, 3), padding='same', activation="relu", strides=1)(p23)
    p24 = keras.layers.Conv2D(f[3], (3,3), padding='same', strides=1)(p24)
    p24=keras.layers.MaxPool2D((2, 2), (2, 2))(p24)
    
    p25 =keras.layers.Conv2D(f[4],(3,3),padding="same",activation="relu",strides=1)(p24)
    p25 =keras.layers.Conv2D(f[4],(3,3),padding="same",activation="relu",strides=1)(p25)
    p25= keras.layers.MaxPool2D((2, 2), (2, 2))(p25)  
    # Fusion
    v1 = keras.layers.Conv2D(f[4], (1, 1), padding='same', strides=1)(p15)
    q2 = keras.layers.Conv2D(f[4], (1, 1), padding='same', strides=1)(p25)
    a1 = keras.layers.MultiHeadAttention(num_heads=8, key_dim=f[4]//8)(q2, v1)
    o1 = keras.layers.LayerNormalization()(a1 + v1)
    fu1 = keras.layers.LayerNormalization()(o1 + keras.layers.Conv2D(f[4], (1, 1), padding='same', strides=1)(o1))

    v2 = keras.layers.Conv2D(f[4], (1, 1), padding='same', strides=1)(p25)
    q1 = keras.layers.Conv2D(f[4], (1, 1), padding='same', strides=1)(p15)
    a2 = keras.layers.MultiHeadAttention(num_heads=8, key_dim=f[4]//8)(q1, v2)
    o2 = keras.layers.LayerNormalization()(a2 + v2)
    fu2 = keras.layers.LayerNormalization()(o2 + keras.layers.Conv2D(f[4], (1, 1), padding='same', strides=1)(o2))
    
    #concatenate
    c = keras.layers.Conv2D(f[5], (3,3), padding='same', strides=(1,3))(keras.layers.Concatenate()([fu1,fu2]))
    c = keras.layers.Activation('relu')(c)
    c = keras.layers.Conv2DTranspose(f[5], (3,3), padding='valid', strides=1)(c)
    c = keras.layers.Activation('relu')(c)
    c = keras.layers.Conv2D(f[5], (3,3), padding='same', strides=1)(c)
    fu = keras.layers.Activation('relu')(c)

    # Decoder
    us1 = keras.layers.UpSampling2D((2, 2))(fu)
    c1 = keras.layers.Conv2D(f[4], (3, 3), padding='same', activation="relu",strides=1)(us1)
    c1 = keras.layers.Conv2D(f[4], (3, 3), padding='same', activation="relu",strides=1)(c1)

    us2 = keras.layers.UpSampling2D((2, 2))(c1)
    c2 = keras.layers.Conv2D(f[3], (3, 3), padding='same', activation="relu",strides=1)(us2)
    c2 = keras.layers.Conv2D(f[3], (3, 3), padding='same', activation="relu",strides=1)(c2)

    us3 = keras.layers.UpSampling2D((2, 2))(c2)
    c3 = keras.layers.Conv2D(f[2], (3, 3), padding='same', activation="relu",strides=1)(us3)
    c3 = keras.layers.Conv2D(f[2], (3, 3), padding='same', activation="relu",strides=1)(c3)

    us4 = keras.layers.UpSampling2D((2, 2))(c3)
    c4 = keras.layers.Conv2D(f[1], (3, 3), padding='same', activation= "relu",strides=1)(us4)
    c4 = keras.layers.Conv2D(f[1], (3, 3), padding='same', activation="relu",strides=1)(c4)
    
    us5 = keras.layers.UpSampling2D((2, 2))(c4)
    c5 = keras.layers.Conv2D(f[0], (3, 3), padding='same', activation= "relu",strides=1)(us5)
    c5 = keras.layers.Conv2D(f[0], (3, 3), padding='same', activation="relu",strides=1)(c5)
    

    outputs = keras.layers.Conv2D(1, (1, 1), padding='same')(c5)
    model = tf.keras.Model([inputs1, inputs2], outputs)

    return model
"""
def connect_block(x, filters, kernel_size=(3,3)):
    c = keras.layers.Conv2D(filters, kernel_size, padding='same', strides=(1,3))(x)
    c = keras.layers.Activation('relu')(c)
    c = keras.layers.Conv2DTranspose(filters, kernel_size, padding='valid', strides=1)(c)
    c = keras.layers.Activation('relu')(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding='same', strides=1)(c)
    c = keras.layers.Activation('relu')(c)
    return c

def metric(optimizer):
    def learning_rate(y_true, y_pred):
        return optimizer.learning_rate
    return learning_rate
"""
model = create_unet_model()
model.summary()
print("model basariyla olusturuldu")
#train
"""
total_epoch = 1
batch_size = 10
model_path = path + sub_path + 'model.h5'
Adam = keras.optimizers.Adam(learning_rate=1e-4)
lr_metric = metric(Adam)
model.compile(optimizer=Adam, loss='mse', metrics=[lr_metric])
model_checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)
lr_checkpoint = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.98, patience=1, min_lr=0)
history = model.fit(x=[train_data1, train_data2], y=train_mask, batch_size=batch_size, epochs=total_epoch, verbose=2, \
    validation_data=([test_data1, test_data2], test_mask), callbacks=[model_checkpoint, lr_checkpoint])

# Testing
model.load_weights(model_path)
model.evaluate(x=[test_data1,test_data2], y=test_mask, batch_size=batch_size)
test_pred = model.predict([test_data1,test_data2])
sio.savemat(path+sub_path+'data1.mat', {'data1': test_data1})
sio.savemat(path+sub_path+'data2.mat', {'data2': test_data2})
sio.savemat(path+sub_path+'mask.mat', {'mask': test_mask})
sio.savemat(path+sub_path+'pred.mat', {'pred': test_pred})

print("model ciktilari basariyla kaydedildi")"""
