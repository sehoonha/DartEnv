import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six

from PyQt4 import QtGui

try:
    import pydart2 as pydart
    from pydart2.gui.viewer import *
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pydart2.)".format(e))

class PydartStaticWindow(PydartWindow):
    def run_application(self,):
        self.show()

def getViewer(sim, title=None, default_camera=None):
    # glutInit(sys.argv)
    win = PydartStaticWindow(sim, title)
    if default_camera is not None:
        win.camera_event(default_camera)
    win.run_application()
    return win

class DartEnv(gym.Env):
    """Superclass for all Dart environments.
        """
    
    def __init__(self, model_path, frame_skip, observation_size, action_bounds, dt=0.002):
        pydart.init()
        print('pydart initialization OK')
        
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist"%fullpath)
    
        self.dart_world = pydart.World(dt, fullpath)
        self.robot_skeleton = self.dart_world.skeletons[-1] # assume that the skeleton of interest is always the last one
        
        for jt in xrange(0, len(self.robot_skeleton.joints)):
            if self.robot_skeleton.joints[jt].has_position_limit(0):
                self.robot_skeleton.joints[jt].set_position_limit_enforced(True)

self.frame_skip= frame_skip
    self.viewer = None
        
        self.metadata = {
        #'render.modes': ['human', 'rgb_array'],
        #'video.frames_per_second' : int(np.round(1.0 / self.dt))
        }
        
        observation, _reward, done, _info = self._step(np.zeros(len(action_bounds[0])))
        assert not done
        self.obs_dim = observation_size
        self.act_dim = len(action_bounds[0])
        
        self.action_space = spaces.Box(action_bounds[1], action_bounds[0])
        
        
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        
        self._seed()
        
        self.viewer = None
        
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : int(np.round(1.0 / self.dt))
}
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    # methods to override:
    # ----------------------------
    def reset_model(self):
        """
            Reset the robot degrees of freedom (qpos and qvel).
            Implement this in each subclass.
            """
        raise NotImplementedError
    
    def viewer_setup(self):
        """
            This method is called when the viewer is initialized and after every reset
            Optionally implement this method, if you need to tinker with camera position
            and so forth.
            """
        pass
    
    # -----------------------------
    
    def _reset(self):
        ob = self.reset_model()
        return ob
    
    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.robot_skeleton.ndofs,) and qvel.shape == (self.robot_skeleton.ndofs,)
        self.robot_skeleton.set_positions(qpos)
        self.robot_skeleton.set_velocities(qvel)
    
    @property
    def dt(self):
        return self.dart_world.dt * self.frame_skip
    
    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

def _render(self, mode='human', close=False):
    if close:
        if self.viewer is not None:
            self._get_viewer().close()
                self.viewer = None
            return
    
        if mode == 'rgb_array':
            self._get_viewer().glwidget.updateGL()
            img = self._get_viewer().glwidget.grabFrameBuffer()
            
            height = img.height()
            width = img.width()
            
            data = np.zeros((height, width, 3))
            print data.shape
            for i in xrange(width):
                for j in xrange(height):
                    color = QtGui.QColor(img.pixel(i, j))
                    rgbval = color.getRgb()
                    data[j][i][0] = rgbval[0]
                    data[j][i][1] = rgbval[1]
                    data[j][i][2] = rgbval[2]

    return data
        elif mode == 'human':
            self._get_viewer().glwidget.updateGL()

def _get_viewer(self):
    if self.viewer is None:
        self.viewer = getViewer(self.dart_world)
        return self.viewer
    
    
    def state_vector(self):
        return np.concatenate([
                               self.robot_skeleton.q,
                               self.robot_skeleton.dq
                               ])