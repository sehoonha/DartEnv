__author__ = 'yuwenhao'

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.core import Layer
import theano.tensor as T, theano
from keras import backend as K
import numpy as np
import copy
import os

import joblib

# WARNING: A lot of hand-coded stuff for now
class DartHopperEnvCont(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # UPOSI variables
        self.use_OSI = False
        self.history_length = 5 # size of the motion history for UPOSI
        self.state_action_buffer = []

        modelpath = os.path.join(os.path.dirname(__file__), "models")
        self.UP = joblib.load(os.path.join(modelpath, 'UP.pkl'))
        if self.use_OSI:
            self.OSI = load_model(os.path.join(modelpath, 'OSI.h5'))
            self.OSI_out = K.function([self.OSI.input, K.learning_phase()], self.OSI.output)

        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)

        if self.use_OSI:
            self.OSI_obs_dim = (self.obs_dim+self.act_dim)*self.history_length+self.obs_dim
            self.obs_dim = 2
            self.observation_space = spaces.Box(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        self.act_dim = 2
        self.action_space = spaces.Box(np.array([0, 0]), np.array([1, 1]))
        self.observation_space = spaces.Box(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))

        utils.EzPickle.__init__(self)

    def _step(self, a):
        if not len(a) == 2:
            action = [0, 0, 0]
        elif self.use_OSI:
            cur_obs = np.hstack([self.state_action_buffer[-1][0], a])
            _, data = self.UP.get_action(cur_obs)
            action = data['mean']
            self.state_action_buffer[-1].append(np.array(action))
        else:
            cur_obs = np.hstack([self._get_obs(), a])
            _, data = self.UP.get_action(cur_obs)
            action = data['mean']

        posbefore = self.model.data.qpos[0,0]
        self.do_simulation(action, self.frame_skip)
        posafter,height,ang = self.model.data.qpos[0:3,0]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus

        reward -= 1e-3 * np.square(action).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (height < 1.8) and (abs(ang) < .4))
        ob = self._get_obs()
        return ob, reward, done, {'vel_rew':(posafter - posbefore) / self.dt, 'action_rew':1e-3 * np.square(action).sum()}

    def _get_obs(self):
        state = np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat,-10,10)
        ])

        if not self.use_OSI:
            return state

        out_ob = np.zeros(self.OSI_obs_dim)
        ind = 0
        for s_a in self.state_action_buffer:
            out_ob[ind:ind+len(s_a[0])] = np.array(s_a[0])
            ind += len(s_a[0])
            out_ob[ind:ind+len(s_a[1])] = np.array(s_a[1])
            ind += len(s_a[1])
        out_ob[ind:ind + len(state)] = np.array(state)

        self.state_action_buffer.append([np.array(state)])
        if len(self.state_action_buffer) > self.history_length:
            self.state_action_buffer.pop(0)
        pred_param = self.OSI_out([[out_ob], 0])[0]

        return pred_param

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.state_action_buffer = [] # for UPOSI
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20