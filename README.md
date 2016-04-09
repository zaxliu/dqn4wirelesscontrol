# DQN for Wireless Control
This projects provides a modular python-based Q learning implementation as well as two example applications in wireless network control.

We implemented both the canonnical table-based agent in [`qlearning.qtable.QAgent`](https://github.com/zaxliu/dqn4wirelesscontrol/blob/master/qlearning/qtable.py) and the more recent neuron-network-based version, a.k.a [deep q network (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), using Theano+Lasagne in [`qlearning.qnn.QAgentNN`](https://github.com/zaxliu/dqn4wirelesscontrol/blob/master/qlearning/qnn.py). These two versions share the same programming interface, which can be handy when one want to start with a simple interpretable agent and then evolve to a more sophisticated one.

For testing, we provide a simple maze example in [`qlearning.simple_env.SimpleMaze`](https://github.com/zaxliu/dqn4wirelesscontrol/blob/master/qlearning/simple_envs.py).

For non-Markovian environments, we also provide a mixin class in [`qlearning.mixin.PhiMixin`](https://github.com/zaxliu/dqn4wirelesscontrol/blob/master/qlearning/mixin.py). It can be used with either type of agents to stack the historical observation into a augmented observation vector. For better exploration and exploitation tradeoff, we implemented a anealing mixin class in [`qlearning.mixin.AnealMixin`](https://github.com/zaxliu/dqn4wirelesscontrol/blob/master/qlearning/mixin.py) for gradually decrease exploration rate.

The two wireless applications are base station sleeping control and transmission scheduling in wireless sensor networks.

