# DQN for Wireless Control
This projects provides a modular python-based deep reinforcement learning implementation as well as an example application in wireless network control.

The package currently implements both the canonnical table-based Q learning agent in [`rl.qtable.QAgent`](https://github.com/zaxliu/dqn4wirelesscontrol/blob/master/qlearning/qtable.py) and the more recent neuron-network-based version, a.k.a [deep q network (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), using Theano and Lasagne in [`rl.qnn_theano.QAgentNN`](https://github.com/zaxliu/dqn4wirelesscontrol/blob/master/qlearning/qnn.py).

A set of additional agent features have also been implemented as Mixin classes in `rl.mixin`.
  1. In non-Markovian environments, `PhiMixin` can be used with either type of agents to stack the historical observation into a augmented observation vector.
  2. For a better exploration and exploitation tradeoff, we implemented a anealing mixin class in `AnealMixin` for gradually decrease exploration rate.
  3. When a environment model is available, `DynaMixin` can be used to incorporate planning into the learnign process.

For testing, we provide a simple maze example in [`rl.simple_env.SimpleMaze`](https://github.com/zaxliu/dqn4wirelesscontrol/blob/master/qlearning/simple_envs.py).

The wireless networking application is dynamic online base station sleeping control.

