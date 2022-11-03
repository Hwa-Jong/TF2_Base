from ast import In
from . import common
from  tensorflow.keras import layers, Model, Input

class BaseModel(Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv1 = common.BasicBlock(32, 3, layers.ReLU())
        self.conv2 = common.BasicBlock(32, 3, layers.ReLU())
        self.conv3 = common.BasicBlock(3, 1, layers.ReLU())

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
