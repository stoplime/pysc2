# Trains a learning algorithm to find shards using a convnet and a form of reinforcement

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
from keras.layers import BatchNormalization, Activation, Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.models import Model

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

class Play(base_agent.BaseAgent):
    """
    This model will use a convnet with input of observations and output of location.
    This location will act as the target location of where the marine should go.
    Training this model will use a form of reinforcement where the loss is calculated based off of the delta reward.
    Each action will be allowed a time frame for the marines to do the action until an evaluation of the delta reward is in place.
    This provides a one to one ratio between actions and reward.
    """

    def modelSetup(self, inputShape, outputShape, hiddenList):
        self.InputTensor = Input(inputShape)
        self.InputShape = inputShape
        self.OutputShape = outputShape
        self.model = Model(self.InputTensor, self.createModel(hiddenList), name='mainModel')
        self.model.compile(loss=self.customLoss, optimizer='adam')
        self.model.summary()

    def createModel(self, hiddenList):
        x = conv2d_bn(self.InputTensor, hiddenList[0][0], hiddenList[0][1], hiddenList[0][1], strides=(hiddenList[0][2], hiddenList[0][2]))
        for i in range(len(hiddenList)-1):
            x = conv2d_bn(x, hiddenList[i][0], hiddenList[i][1], hiddenList[i][1], strides=(hiddenList[i][2], hiddenList[i][2]))
        x = Flatten()(x)
        return x

    def customLoss(self, y_true, y_pred):
        """
        Will not use y_true since it will use a reward setup instead
        """
        
        return cce

    def setup(self, obs_spec, action_spec):
        super(Play, self).setup(obs_spec, action_spec)
        self.total_actions = 0
        self.arg_sizes = []
        print("action_spec.FUNCTIONS type", len(action_spec.functions))
        for action, val in enumerate(action_spec.functions):
            self.total_actions += 1
            for arg in val.args:
                self.arg_sizes.append(arg.sizes)
                for size in arg.sizes:
                    self.total_actions += 1
        self.modelSetup(inputShape=(84,84,20), outputShape=(2,), hiddenList=[(32, 8, 4), (64, 4, 2), (64, 3, 1), (512, 1, 1), (2, 1, 1)])

    def step(self, obs):
        super(Play, self).step(obs)
        # print(obs.observation["score_cumulative"])
        # print("screen", obs.observation["screen"].shape)
        # print("minimap", obs.observation["minimap"].shape)
        # plt.imshow(obs.observation["screen"][1,:,:])

        function_id = np.random.choice(obs.observation["available_actions"])
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)
