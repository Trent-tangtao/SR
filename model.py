import tensorflow as tf
from keras.layers import Conv2D,ReLU,Input,Activation
from keras.models import Model


def SRCNN(width=33,height=33,depth=1):
    input = Input(shape=(height,width,depth))
    conv1 = Conv2D(64,kernel_size=9,use_bias=True,kernel_initializer='he_uniform',)(input)
    relu = Activation('relu')(conv1)
    conv2 = Conv2D(32,kernel_size=1,use_bias=True,kernel_initializer='he_uniform',)(relu)
    relu = Activation('relu')(conv2)
    conv3= Conv2D(1,kernel_size=5,use_bias=True,kernel_initializer='he_uniform',)(relu)
    model = Model(inputs=input,outputs=conv3)
    return model


# 另一种思路
def predict_SRCNN():
    input = Input(shape=(None, None, 1))
    conv1 = Conv2D(64,kernel_size=9,use_bias=True,kernel_initializer='he_uniform',)(input)
    relu = Activation('relu')(conv1)
    conv2 = Conv2D(32,kernel_size=1,use_bias=True,kernel_initializer='he_uniform',)(relu)
    relu = Activation('relu')(conv2)
    conv3= Conv2D(1,kernel_size=5,use_bias=True,kernel_initializer='he_uniform',)(relu)
    model = Model(inputs=input,outputs=conv3)
    return model