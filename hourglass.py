import os

from keras.losses import mean_squared_error
from keras.optimizers import Adam, RMSprop

from hourglass_blocks import create_hourglass_network, bottleneck_block
