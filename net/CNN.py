import os
from keras.layers import Input, Conv2D, ReLU, LeakyReLU, BatchNormalization, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, UpSampling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback, TensorBoard
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import Adam
from keras import backend
 
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def conv_block(x_in,  nf, kernel_size=3, use_bn=False):
    x_out = Conv2D(nf, (kernel_size, kernel_size), padding='same')(x_in)
    x_out = LeakyReLU(0.1)(x_out)  
    if use_bn:
        x_out = BatchNormalization()(x_out)
      
    return x_out
  
def conv_block_ReLu(x_in,  nf, kernel_size=3, use_bn=False):
    x_out = Conv2D(nf, (kernel_size, kernel_size), padding='same')(x_in)
    x_out = ReLU()(x_out)  
    if use_bn:
        x_out = BatchNormalization()(x_out)
      
    return x_out
  
def downLayer(inputLayer, nf1, nf2, bn=False):
    conv = conv_block(inputLayer, nf1, kernel_size=3, use_bn=bn)
    conv = conv_block(conv,       nf2, kernel_size=3, use_bn=bn)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)

    return pool, conv
  
def downLayer_new(inputLayer, nf, bn=False):
    conv = conv_block(inputLayer, nf, kernel_size=3, use_bn=bn)
    conv = conv_block(conv,       nf, kernel_size=3, use_bn=bn)
    pool =  MaxPooling2D(pool_size=(2, 2))(conv)

    return pool, conv

# upsampling, synthesis path
def upLayer_new(inputLayer, concatLayer, nf, bn=False, do= 0.0):
    up = Conv2DTranspose(nf*2, (2, 2), strides=(2, 2), padding='same')(inputLayer)
    up = LeakyReLU(0.1)(up)  
    up = concatenate([up, concatLayer])

    conv = conv_block(up, nf, kernel_size=3, use_bn=bn)
    conv = Dropout(do)(conv)
    conv = conv_block(conv, nf, kernel_size=3, use_bn=bn)
    return conv

def make_model(hps):  
  if hps['cnn_depht'] == 2:
    return make_model_d2(hps)
  elif hps['cnn_depht'] == 3:
    return make_model_d3(hps)
  else:
    return -1
    
def make_model_d2(hps):  
  sfs           = hps['sfs']
  bn            = hps['bn']
  do            = hps['do']
  learing_rate  = hps['LR']
  x             = hps['netInput_x']
  y             = hps['netInput_y']
  input = Input((x, y, 3))
  conv1, conv1_b_m   = downLayer(input, sfs,   sfs*2,  bn) ### 128 x 128 -> 64 x 64
  conv2, conv2_b_m   = downLayer(conv1, sfs*2, sfs*4,  bn) ### -> 32 x 32
#   conv3, conv3_b_m   = downLayer_new(conv2, sfs*4,  bn) ### 4 x 4
#   conv4, conv4_b_m   = downLayer_new(conv3, sfs*8, bn) ### 2 x 2
#   conv5, conv5_b_m   = downLayer_new(conv4, sfs*16, bn) ### 2 x 2

  conv_bottom = conv_block(conv2,       sfs*4,  kernel_size=3,  use_bn=bn) ### 32 x 32
  conv_bottom = conv_block(conv_bottom, sfs*8, kernel_size=3,  use_bn=bn) ### 32 x 32


#   conv_up5  = upLayer_new(conv_bottom, conv5_b_m, sfs*32, bn, do) ### 4 x 4
#   conv_up4  = upLayer_new(conv_bottom,    conv4_b_m, sfs*16, bn, do) ### 8 x 8
#   conv_up3  = upLayer_new(conv_up4,    conv3_b_m, sfs*8,  bn, do) ### 16 x 16
  conv_up2  = upLayer_new(conv_bottom, conv2_b_m, sfs*4,  bn, do) ### 32 x 32
  conv_up1  = upLayer_new(conv_up2,    conv1_b_m, sfs*2,  bn, do) ### 64 x 64

  conv_last = conv_block(conv_up1, 1, kernel_size=1)
  # conv_last = Conv2D(1, (1, 1), padding='same')(conv_up1)

  model = Model(inputs=input, outputs=conv_last)
  model.compile(optimizer=Adam(lr=learing_rate), loss = 'mse', metrics = ['mse'])
  return model
  
def make_model_d3(hps):   
  sfs           = hps['sfs']
  bn            = hps['bn']
  do            = hps['do']
  learing_rate  = hps['LR']
  x             = hps['netInput_x']
  y             = hps['netInput_y']

  input = Input((x, y, 3))
  conv1, conv1_b_m  = downLayer(input, sfs,   sfs*2,  bn) ### 128 x 128 -> 64 x 64
  conv2, conv2_b_m  = downLayer(conv1, sfs*2, sfs*4,  bn) ### 64 x 64 -> 32 x 32
  conv3, conv3_b_m  = downLayer(conv2, sfs*4, sfs*8,  bn) ### 32 x 32 -> 16 x 16

  conv_bottom = conv_block(conv3,       sfs*8 , kernel_size=3,  use_bn=bn) ### 16 x 16 -> 16 x 16
  conv_bottom = conv_block(conv_bottom, sfs*16 , kernel_size=3,  use_bn=bn) ### 16 x 16 -> 16 x 16

  conv_up3 = upLayer_new(conv_bottom, conv3_b_m, sfs*8,  bn, do) ### 16 x 16 -> 32 x 32
  conv_up2 = upLayer_new(conv_up3,    conv2_b_m, sfs*4,  bn, do) ### 32 x 32 -> 64 x 64
  conv_up1 = upLayer_new(conv_up2,    conv1_b_m, sfs*2,  bn, do) ### 64 x 64 -> 128 x 128

  conv_last = conv_block(conv_up1, 1, kernel_size=1)
  # conv_last = Conv2D(1, (1, 1), padding='same')(conv_up1)

  model = Model(inputs=input, outputs=conv_last)
  model.compile(optimizer=Adam(lr=learing_rate), loss = 'mse', metrics = ['mse', 'mae', rmse, 'mape', 'cosine'])
  return model

def make_model_d2_feature_jump(hps):  
  sfs           = hps['sfs']
  bn            = hps['bn']
  do            = hps['do']
  learing_rate  = hps['LR']
  x             = hps['netInput_x']
  y             = hps['netInput_y']
  input = Input((x, y, 3))
  conv1, conv1_b_m   = downLayer_new(input, sfs,    bn) ### 32 x 32
  conv2, conv2_b_m   = downLayer_new(conv1, sfs*2,  bn) ### 16 x 16
#   conv3, conv3_b_m   = downLayer_new(conv2, sfs*4,  bn) ### 4 x 4
#   conv4, conv4_b_m   = downLayer_new(conv3, sfs*8, bn) ### 2 x 2
#   conv5, conv5_b_m   = downLayer_new(conv4, sfs*16, bn) ### 2 x 2

  conv_bottom = conv_block(conv2, sfs*32 , kernel_size=3,  use_bn=bn) ### 2 x 2
  conv_bottom = conv_block(conv_bottom, sfs*32 , kernel_size=3,  use_bn=bn) ### 2 x 2


#   conv_up5  = upLayer_new(conv_bottom, conv5_b_m, sfs*32, bn, do) ### 4 x 4
#   conv_up4  = upLayer_new(conv_bottom,    conv4_b_m, sfs*16, bn, do) ### 8 x 8
#   conv_up3  = upLayer_new(conv_up4,    conv3_b_m, sfs*8,  bn, do) ### 16 x 16
  conv_up2  = upLayer_new(conv_bottom,    conv2_b_m, sfs*4,  bn, do) ### 32 x 32
  conv_up1  = upLayer_new(conv_up2,    conv1_b_m, sfs*2,  bn, do) ### 64 x 64

  conv_last = conv_block(conv_up1, 1, kernel_size=1)

  model = Model(inputs=input, outputs=conv_last)
  model.compile(optimizer=Adam(lr=learing_rate), loss = 'mse', metrics = ['mse', 'mae', rmse, 'mape', 'cosine'])
  return model
  
def make_model_d3_feature_jump(hps):   
  sfs           = hps['sfs']
  bn            = hps['bn']
  do            = hps['do']
  learing_rate  = hps['LR']
  x             = hps['netInput_x']
  y             = hps['netInput_y']

  input = Input((x, y, 3))
  conv1, conv1_b_m   = downLayer_new(input, sfs,    bn) ### 32 x 32
  conv2, conv2_b_m   = downLayer_new(conv1, sfs*2,  bn) ### 16 x 16
  conv3, conv3_b_m   = downLayer_new(conv2, sfs*4,  bn) ### 4 x 4
#   conv4, conv4_b_m   = downLayer_new(conv3, sfs*8, bn) ### 2 x 2
#   conv5, conv5_b_m   = downLayer_new(conv4, sfs*16, bn) ### 2 x 2

  conv_bottom = conv_block(conv3, sfs*32 , kernel_size=3,  use_bn=bn) ### 2 x 2
  conv_bottom = conv_block(conv_bottom, sfs*32 , kernel_size=3,  use_bn=bn) ### 2 x 2


#   conv_up5  = upLayer_new(conv_bottom, conv5_b_m, sfs*32, bn, do) ### 4 x 4
#   conv_up4  = upLayer_new(conv_bottom,    conv4_b_m, sfs*16, bn, do) ### 8 x 8
  conv_up3  = upLayer_new(conv_bottom,    conv3_b_m, sfs*8,  bn, do) ### 16 x 16
  conv_up2  = upLayer_new(conv_up3,    conv2_b_m, sfs*4,  bn, do) ### 32 x 32
  conv_up1  = upLayer_new(conv_up2,    conv1_b_m, sfs*2,  bn, do) ### 64 x 64

  conv_last = conv_block(conv_up1, 1, kernel_size=1)

  model = Model(inputs=input, outputs=conv_last)
  model.compile(optimizer=Adam(lr=learing_rate), loss = 'mse', metrics = ['mse', 'mae', rmse, 'mape', 'cosine'])
  print('model metrics: ', model.metrics_names)
  return model