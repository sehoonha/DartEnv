import numpy as np
from gym import utils
from gym.envs.dart import dart_env

class DartCartPoleEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.actions = list()
        control_bounds = np.array([[10.0],[-10.0]])
        self.action_scale = 1.0
        dart_env.DartEnv.__init__(self, 'cartpole.skel', 2, 4, control_bounds,
                                  dt=0.01)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        # self.actions.append(a)
        # print("a = %.4f" % a[0])
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = (a[0] - 0.5) * self.action_scale

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone

        if done:
            avg = None if len(self.actions) == 0 else np.mean(self.actions)
            # print("%d actions: avg = %s" % (len(self.actions), avg))
            self.actions = list()

        # reward = 1.0
        notdone = 1 - int(done)
        ucost = 1e-5 * (a ** 2).sum()
        xcost = 1 - np.cos(ob[1])
        # xcost2 = 1e-2 * (ob[0] ** 2)
        reward = notdone * 10 - notdone * (xcost) - notdone * ucost
        # print(ob)

        return ob, reward, done, {}

    def log_diagnostics(self, paths):
        pass

    def terminate(self, ):
        pass

    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

    def reset_model(self):
        self.actions = list()
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
