# Test run for reinforcement learning

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
# import keras

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


class Play(base_agent.BaseAgent):
    """An agent designed for general play. Based off of learned random"""

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
        # print("self.total_actions: ", self.total_actions)
        # print("self.arg_sizes: ", self.arg_sizes)

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
