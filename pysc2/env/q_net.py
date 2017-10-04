# q_net is the quality assesment of the environment
# it generalizes the game into a simple network that is diferentiable

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.core import Dropout

from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19 
from keras.applications.vgg16 import VGG16

class qnet(object):
    # initializes a new qnet or load one from json
    def __init__(self, load_json=None):
        self.input_tensor = 
        if load_json == None:
            self.qnet_model = self.create_model()

    def create_model(self, model_type='inceptionv3', load_weights=None):
        
