# DartEnv
Openai Gym with Dart support

###About

**DartEnv** is an implementation of the <a href="https://github.com/openai/gym">**openai gym**</a> environment.

It is currently still under development.

###Requirements

You need to install these packages first:

<a href="http://dartsim.github.io/">Dart</a>

<a href="http://pydart2.readthedocs.io/en/latest/">PyDart2</a>

###Install

To facilitate installation, we have uploaded the entire project base including the original openai gym code. To install, simply do 


    git clone https://github.com/VincentYu68/DartEnv.git
	  cd gym
	  pip install -e .


###Example

After installation, you can run DartEnv using the same API as openai gym. One example of running the dart version of the Hopper model is shown below:

    import gym
    env = gym.make('DartHopper-v1')
    env.reset()
    env.render()

