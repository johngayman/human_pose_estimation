import keras.backend as K
from keras.layers import *
from keras.models import *

### Ref: https://github.com/robertklee/COCO-Human-Pose/blob/main/hourglass_blocks.py

def create_hourglass_network(num_classes, num_stacks, num_channels, inres, bottleneck, activation_str):
  #shoudnt use the worsed
  input = Input(shape = (inres[0], inres[1], 3))

  front_features = create_front_module(input, num_channels, bottleneck)

  head_next_stage = front_features

  outputs = []

  for i in range(num_stacks):
    head_next_stage, head_to_loss = hourglass_module(head_next_stage, num_classes, num_channels, bottleneck, i, activation_str)
    outputs.append(head_to_loss)

  model = Model(inputs = input, outputs = outputs)
  
  return model

def hourglass_module(input, num_classes, num_channels, bottleneck, hgid, activation_str):
  ## Create left half features
  # lf1, lf2, lf4, lf8: 1, 1/2, 1/4, 1/8 resolution 
  #  every maxpool res reduced by 2, until 4x4 (64/16)
  
  hgname = 'hg' + str(hgid)

  lf1 = bottleneck(input, num_channels, hgname + '_lf1_bottleneck') 
  _x = MaxPool2D(pool_size = (2, 2),  strides = (2, 2), name = hgname + 'max_pool1')(lf1)
  
  lf2 = bottleneck(_x, num_channels, hgname + '_lf2_bottleneck')
  _x = MaxPool2D(pool_size = (2, 2),  strides = (2, 2), name = hgname + '_max_pool2')(lf2)
  
  lf4 = bottleneck(_x, num_channels, hgname + '_lf4_bottleneck')
  _x = MaxPool2D(pool_size = (2, 2),  strides = (2, 2), name = hgname + '_max_pool4')(lf4)
  
  lf8 = bottleneck(_x, num_channels, hgname + '_lf8_bottleneck')
  _x = MaxPool2D(pool_size = (2, 2),  strides = (2, 2), name = hgname + '_max_pool8')(lf8)


  ## Bottom layer
  _x = bottleneck(_x, num_channels, hgname + '_bottom_1')
  _x = bottleneck(_x, num_channels, hgname + '_bottom_2')
  _x = bottleneck(_x, num_channels, hgname + '_bottom_3')

  
  ## Skip layer
  skip8 = bottleneck(lf8, num_channels, hgname + '_skip8')
  skip4 = bottleneck(lf4, num_channels, hgname + '_skip4')
  skip2 = bottleneck(lf2, num_channels, hgname + '_skip2')
  skip1 = bottleneck(lf1, num_channels, hgname + '_skip1')


  ## Right half feature
  # every upsamply res increased by 2, starting from 4x4 (64/16)
  rf8 = UpSampling2D(name = hgname + '_up_scale8')(_x) #bottom layer output
  rf8 = Add(name = hgname + '_add8')([skip8, rf8])
  rf8 = bottleneck(rf8, num_channels, hgname + '_rf8_bottleneck')

  rf4 = UpSampling2D(name = hgname + 'up_scale4')(rf8)
  rf4 = Add(name = hgname + '_add4')([skip4, rf4])
  rf4 = bottleneck(rf4, num_channels, hgname + '_rf4_bottleneck')

  rf2 = UpSampling2D(name = hgname + '_up_scale2')(rf4)
  rf2 = Add(name = hgname + '_add2')([skip2, rf2])
  rf2 = bottleneck(rf2, num_channels, hgname + '_rf2_bottleneck')

  rf1 = UpSampling2D(name = hgname + '_up_scale1')(rf2)
  rf1 = Add(name = hgname + '_add1')([skip1, rf1])
  rf1 = bottleneck(rf1, num_channels, hgname + '_rf1_bottleneck')

  head_to_next_stage, head_to_loss = create_heads(input, rf1, num_classes, hgid, num_channels, activation_str)

  return head_to_next_stage, head_to_loss

def bottleneck_block(input, num_out_channels, block_name):
  ## Skip layer if the num of channels of input == num_out_channels then no need to map
  if K.int_shape(input)[-1] == num_out_channels:
    _skip = input
  else:
    _skip = Conv2D(filters = num_out_channels, kernel_size = (1, 1), activation = 'relu', padding = 'same', 
                   name = block_name + '_skip')(input)

  ## Residual: 3 Conv blocks, [num_out_channels/2 ->  num_out_channels/2 -> num_out_channels]
  _x = Conv2D(filters = num_out_channels // 2, kernel_size = (1, 1), activation = 'relu', padding = 'same', 
              name = block_name + '_conv_1_1x1')(input)
  _x = BatchNormalization(name = block_name + '_batch_norm_1')(_x)
  _x = Conv2D(filters = num_out_channels // 2, kernel_size = (3, 3), activation = 'relu', padding = 'same', 
              name = block_name + '_conv_2_3x3')(_x)
  _x = BatchNormalization(name = block_name + '_batch_norm_2')(_x)
  _x = Conv2D(filters = num_out_channels, kernel_size = (1, 1), activation = 'relu', padding = 'same', 
              name = block_name + '_conv_3_1x1')(_x)
  _x = BatchNormalization(name = block_name + '_batch_norm_3')(_x)

  ## Add
  _x = Add(name = block_name + '_add')([_skip, _x])
  
  return _x

def create_front_module(input, num_channels, bottleneck):
  ## Front module
  # 1 7x7 Conv2D + max pool => 1/4 resolution
  # 3 bottleneck blocks
  
  _x = Conv2D(filters = num_channels // 4, kernel_size = (7, 7), strides = (2, 2), activation = 'relu', padding = 'same', 
              name =  'front_conv_1_7x7')(input)
  _x = BatchNormalization(name = 'front_batch_norm_1')(_x)
  _x = bottleneck(_x, num_channels // 2, 'front_bottleneck_1')
  _x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = 'front_maxpool_1')(_x)
  _x = bottleneck(_x, num_channels // 2, 'front_bottleneck_2')
  _x = bottleneck(_x, num_channels, 'front_bottle_neck_2')

  return _x

def create_heads(prev_hg_features, rf1, num_classes, hgid, num_channels, activation_intermediate):
  # two heads, 1 to the next hourglass/stage, 1 to intermediate supervision/loss
  head = Conv2D(num_channels, kernel_size = (1, 1), activation = 'relu', padding = 'same', 
                name = str(hgid) + '_head_conv_1_1x1')(rf1)
  head = BatchNormalization(name = str(hgid) + '_head_batch_norm')(head)

  # head as intermediate supervision, use 'linear' or 'sigmoid' as activation
  head_to_loss = Conv2D(num_classes, kernel_size = (1, 1), activation = activation_intermediate, padding = 'same',
                      name = str(hgid) + '_head_loss_conv_2_1x1_parts')(head)
  # for the next stage
  head = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                name = str(hgid) + '_head_conv_3_1x1')(head)
  head_m = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                name = str(hgid) + '_head_conv_4_1x1')(head_to_loss)
  head_next_stage = Add(name = str(hgid) + '_add')([head, head_m, prev_hg_features])

  return head_next_stage, head_to_loss