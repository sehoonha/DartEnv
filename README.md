# DartEnv
Openai Gym with Dart support

###About

**DartEnv** provides 3D multi-body simulation environments based on the <a href="https://github.com/openai/gym">**openai gym**</a> environment. The physics simulation is carried out by <a href="http://dartsim.github.io/">Dart</a> and <a href="http://pydart2.readthedocs.io/en/latest/">PyDart2</a>, which is a python binding for Dart.

###Requirements

You need to install these packages first:

<a href="http://dartsim.github.io/">Dart</a>

<a href="http://pydart2.readthedocs.io/en/latest/">PyDart2</a>

###Install

To facilitate installation, we have uploaded the entire project base including the original openai gym code. To install, simply do 


    git clone https://github.com/VincentYu68/DartEnv.git
    cd DartEnv
    pip install -e .


###Example

After installation, you can run DartEnv using the same API as openai gym. One example of running the dart version of the Hopper model is shown below:

    import gym
    env = gym.make('DartHopper-v1')
    env.reset()
    env.render()

###Learning with <a href="https://github.com/openai/rllab">RLLAB</a> (As of Feb 19th, 2017)

In order to use RLLAB to learn policies for DartEnv (or other openai gym environments with the current gym version) with video log, a few changes need to be made in RLLAB code.

In rllab.envs.gym_env.py:

Change

    force_reset=False
to

    force_reset=True
    
And change

    recorder = self.env._monitor.stats_recorder
    
to

    recorder = self.env.stats_recorder
