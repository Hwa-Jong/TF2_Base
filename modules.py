import tensorflow as tf
from  tensorflow.keras import layers


class BasicBlock(layers.Layer):
    def __init__(self, filters, kernel_size, act=None, strides=(1, 1), bn=False, bias=True): 
        super(BasicBlock, self).__init__()
        self.conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")

        self.bn = None
        if bn:
            self.bn = layers.BatchNormalization()

        self.act = act


    def call(self, x, training=None, mask=None):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x, training=training)
        if self.act is not None:
            x = self.act(x)        
        return x