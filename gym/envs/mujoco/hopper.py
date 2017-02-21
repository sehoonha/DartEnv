import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # UPOSI variables
        self.use_UPOSI = False
        self.history_length = 3 # size of the motion history for UPOSI
        self.state_action_buffer = []

        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

    def setUseUPOSI(self, useUPOSI = True):
        self.use_UPOSI = useUPOSI
        self.OSI_obs_dim = (self.obs_dim+self.act_dim)*self.history_length+self.obs_dim

    def _step(self, a):
        if self.use_UPOSI:
            self.state_action_buffer[-1].append(np.array(a))

        posbefore = self.model.data.qpos[0,0]
        self.do_simulation(a, self.frame_skip)
        posafter,height,ang = self.model.data.qpos[0:3,0]
        alive_bonus = 0.5
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus

        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {'vel_rew':(posafter - posbefore) / self.dt, 'action_rew':1e-3 * np.square(a).sum()}

    def _get_obs(self):
        state = np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat,-10,10)
        ])

        if self.use_UPOSI:
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

            return np.array([out_ob], dtype=np.float32)

        return state

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
