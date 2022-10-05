from ast import In
import modules
from  tensorflow.keras import layers, Model, Input

class BaseModel(Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv1 = modules.BasicBlock(32, 3, layers.ReLU())
        self.conv2 = modules.BasicBlock(32, 3, layers.ReLU())
        self.conv3 = modules.BasicBlock(3, 1, layers.ReLU())


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


'''
class BaseModel(layers.Layer):
    def __init__(self): 
        super(BaseModel, self).__init__()
        self.istrain = True
        self.conv1 = modules.BasicBlock(32, 3, layers.ReLU())
        self.conv2 = modules.BasicBlock(32, 3, layers.ReLU())

    def train(self):
        self.istrain = True

    def eval(self):
        self.istrain = False

    def call(self, x, training=None, **kwargs):
        x = self.input(x)
        x = self.conv1(x, self.istrain)
        x = self.conv2(x, self.istrain)
        return x


'''