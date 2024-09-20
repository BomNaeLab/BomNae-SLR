import tensorflow as tf
import keras
from keras import Model
from keras import layers



#------------------------------------------------------------------------------------#
##  THIS IS A TRAINING MODEL, IMPLEMENTION DETAIL MAY DIFFER FROM PRODUCTION MODEL  ##
#------------------------------------------------------------------------------------#


# TODO
# THIS IS AN ABSTRACT BACKBONE SCRIPT, ADD ACTUAL IMPLEMENTATION LATER

# NOTE
# model detail
#
#
#

# global variables
model = None # public


# custom layer definition
class Conv2Plus1D(layers.Layer):
    def __init__():
        super().__init__()
        pass

    def call(self, x):
        pass


# the main model class
class SLR_model(Model):
    def __init__(self):
        super().__init__()
        pass

    def call(self, x):
        pass


# the part that doesnt get executed when imported
if __name__ == "__main__":
    pass


def get_model():
    return model

def set_model(new_model):
    global model
    model = new_model

def train_or_fit():
    pass


