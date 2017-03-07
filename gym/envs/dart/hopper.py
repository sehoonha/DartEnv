import numpy as np
from gym import utils
from gym.envs.dart import dart_env

class hopperContactManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.param_dim = 1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        return np.array([friction_param])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

    def resample_parameters(self):
        x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)

class hopperContactMassManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.torso_mass_range = [3.0, 6.0]
        self.param_dim = 2

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        return np.array([friction_param, mass_param])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

        mass = x[1] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

    def resample_parameters(self):
        #x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        x = np.random.normal(0, 0.2, 2) % 1
        self.set_simulator_parameters(x)

class hopperContactMassRoughnessManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.torso_mass_range = [3.0, 6.0]
        self.roughness_range = [-0.05, -0.02] # height of the obstacles
        self.param_dim = 3


    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        cq = self.simulator.dart_world.skeletons[0].q
        cur_height = cq[10]
        roughness_param = (cur_height - self.roughness_range[0]) / (self.roughness_range[1] - self.roughness_range[0])

        return np.array([friction_param, mass_param, roughness_param])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

        mass = x[1] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        obs_height = x[2] * (self.roughness_range[1] - self.roughness_range[0]) + self.roughness_range[0]
        cq = self.simulator.dart_world.skeletons[0].q
        cq[10] = obs_height
        cq[16] = obs_height
        cq[22] = obs_height
        cq[28] = obs_height
        cq[34] = obs_height
        cq[40] = obs_height
        self.simulator.dart_world.skeletons[0].q = cq

    def resample_parameters(self):
        #x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        x = np.random.normal(0, 0.2, self.param_dim) % 1
        self.set_simulator_parameters(x)

        if len(self.simulator.dart_world.skeletons[0].bodynodes) >= 7:
            cq = self.simulator.dart_world.skeletons[0].q
            cq[9] = np.random.random()-0.5
            cq[15] = np.random.random()-0.5
            cq[21] = np.random.random()-0.5
            cq[27] = np.random.random()-0.5
            cq[33] = np.random.random()-0.5
            cq[39] = np.random.random()-0.5
            self.simulator.dart_world.skeletons[0].q = cq

class hopperContactAllMassManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.mass_range = [-1.0, 1.0]
        self.param_dim = 2
        self.initial_mass = []
        for i in range(4):
            self.initial_mass.append(simulator.robot_skeleton.bodynodes[2+i].m)

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass1 = self.simulator.robot_skeleton.bodynodes[2].m - self.initial_mass[0]
        mass_param1 = (cur_mass1 - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])
        cur_mass2 = self.simulator.robot_skeleton.bodynodes[3].m - self.initial_mass[1]
        mass_param2 = (cur_mass2 - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])
        cur_mass3 = self.simulator.robot_skeleton.bodynodes[4].m - self.initial_mass[2]
        mass_param3 = (cur_mass3 - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])
        cur_mass4 = self.simulator.robot_skeleton.bodynodes[5].m - self.initial_mass[3]
        mass_param4 = (cur_mass4 - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])

        return np.array([friction_param, mass_param1])#, mass_param2, mass_param3, mass_param4])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

        mass = x[1] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0] + self.initial_mass[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        '''mass = x[2] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0] + self.initial_mass[1]
        self.simulator.robot_skeleton.bodynodes[3].set_mass(mass)

        mass = x[3] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0] + self.initial_mass[2]
        self.simulator.robot_skeleton.bodynodes[4].set_mass(mass)

        mass = x[4] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0] + self.initial_mass[3]
        self.simulator.robot_skeleton.bodynodes[5].set_mass(mass)'''

    def resample_parameters(self):
        #x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        x = np.random.normal(0, 0.2, 2) % 1
        self.set_simulator_parameters(x)

class hopperContactMassFootUpperLimitManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.torso_mass_range = [3.0, 6.0]
        self.limit_range = [-0.2, 0.2]
        self.param_dim = 2+1
        self.initial_up_limit = self.simulator.robot_skeleton.joints[-3].position_upper_limit(0)

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        # use upper limit of
        limit_diff1 = self.simulator.robot_skeleton.joints[-3].position_upper_limit(0) - self.initial_up_limits
        limit_diff1 = (limit_diff1 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])

        return np.array([friction_param, mass_param, limit_diff1])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

        mass = x[1] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        limit_diff1 = x[2] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_up_limits

        self.simulator.robot_skeleton.joints[-3].set_position_upper_limit(0, limit_diff1)

    def resample_parameters(self):
        #x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        x = np.random.normal(0, 0.2, 2) % 1
        self.set_simulator_parameters(x)

class hopperContactMassAllLimitManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.torso_mass_range = [3.0, 6.0]
        self.limit_range = [-0.3, 0.3]
        self.param_dim = 2+4
        self.initial_up_limits = []
        self.initial_low_limits = []
        for i in range(3):
            self.initial_up_limits.append(simulator.robot_skeleton.joints[-3+i].position_upper_limit(0))
            self.initial_low_limits.append(simulator.robot_skeleton.joints[-3+i].position_lower_limit(0))

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        # use upper limit of
        limit_diff1 = self.simulator.robot_skeleton.joints[-3].position_upper_limit(0) - self.initial_up_limits[0]
        limit_diff2 = self.simulator.robot_skeleton.joints[-2].position_upper_limit(0) - self.initial_up_limits[1]
        limit_diff3 = self.simulator.robot_skeleton.joints[-1].position_upper_limit(0) - self.initial_up_limits[2]
        limit_diff4 = self.simulator.robot_skeleton.joints[-1].position_lower_limit(0) - self.initial_low_limits[2]
        limit_diff1 = (limit_diff1 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])
        limit_diff2 = (limit_diff2 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])
        limit_diff3 = (limit_diff3 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])
        limit_diff4 = (limit_diff4 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])

        return np.array([friction_param, mass_param, limit_diff1, limit_diff2, limit_diff3, limit_diff4])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

        mass = x[1] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        limit_diff1 = x[2] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_up_limits[0]
        limit_diff2 = x[3] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_up_limits[1]
        limit_diff3 = x[4] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_up_limits[2]
        limit_diff4 = x[5] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_low_limits[2]

        self.simulator.robot_skeleton.joints[-3].set_position_upper_limit(0, limit_diff1)
        self.simulator.robot_skeleton.joints[-2].set_position_upper_limit(0, limit_diff2)
        self.simulator.robot_skeleton.joints[-1].set_position_upper_limit(0, limit_diff3)
        self.simulator.robot_skeleton.joints[-1].set_position_lower_limit(0, limit_diff4)

    def resample_parameters(self):
        #x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        x = np.random.normal(0, 0.2, 2) % 1
        self.set_simulator_parameters(x)

class DartHopperEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        self.action_scale = 200
        self.train_UP = True
        self.noisy_input = False
        self.resample_MP = True  # whether to resample the model paraeters
        obs_dim = 11
        self.param_manager = hopperContactMassRoughnessManager(self)
        if self.train_UP:
            obs_dim += self.param_manager.param_dim

        # UPOSI variables
        self.use_UPOSI = False
        self.history_length = 5 # size of the motion history for UPOSI
        self.state_action_buffer = []

        if self.use_UPOSI:
            self.OSI_obs_dim = (obs_dim+len(self.control_bounds[0]))*self.history_length+obs_dim
            obs_dim = self.OSI_obs_dim

        dart_env.DartEnv.__init__(self, 'hopper_obs.skel', 4, obs_dim, self.control_bounds)


        utils.EzPickle.__init__(self)

    def setUseUPOSI(self, useUPOSI = True):
        self.use_UPOSI = useUPOSI
        self.OSI_obs_dim = (self.obs_dim+self.act_dim)*self.history_length+self.obs_dim

    def _step(self, a):
        if self.use_UPOSI and len(self.state_action_buffer) > 0:
            self.state_action_buffer[-1].append(np.array(a))
        pre_state = [self.state_vector()]
        if self.train_UP:
            pre_state.append(self.param_manager.get_simulator_parameters())

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        posbefore = self.robot_skeleton.q[0]
        self.do_simulation(tau, self.frame_skip)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        joint_limit_penalty = 0
        for j in [-2]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward -= 5e-1 * joint_limit_penalty
        #reward -= 1e-7 * total_force_mag

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (height < 1.8) and (abs(ang) < .4))
        ob = self._get_obs()

        if len(self.dart_world.skeletons[0].bodynodes) >= 7: # move obstacles with the hopper
            cq = self.dart_world.skeletons[0].q
            cq[9] += posafter - posbefore
            cq[15] += posafter - posbefore
            cq[21] += posafter - posbefore
            cq[27] += posafter - posbefore
            cq[33] += posafter - posbefore
            cq[39] += posafter - posbefore
            self.dart_world.skeletons[0].q = cq

        return ob, reward, done, {'pre_state':pre_state, 'vel_rew':(posafter - posbefore) / self.dt, 'action_rew':1e-3 * np.square(a).sum(), 'forcemag':1e-7*total_force_mag, 'done_return':done}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10)
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

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

        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])

        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        if self.resample_MP:
            self.param_manager.resample_parameters()

        self.state_action_buffer = [] # for UPOSI

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5