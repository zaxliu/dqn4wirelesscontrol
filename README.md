# DQN for Wireless Control
This projects provides a modular python-based Q learning implementation as well as two example applications in wireless network control.

We implemented both the canonnical table-based agent in qlearning.qtable.QAgent and the more recent neuron-network-based version, a.k.a deep q network (DQN), using Theano+Lasagne in qlearning.qnn.QAgentNN. These two versions share the same programming interface, which can be handy when one want to start with a simple interpretable agent and then evolve to a more sophisticated one.

For non-Markovian environments, we also provide a mixin class in qlearning.mixin.PhiMixin. It can be used with either type of agents to stack the historical observation into a augmented observation vector. For better exploration and exploitation tradeoff, we implemented a anealing mixin class in qlearning.mixin.AnealMixin for gradually decrease exploration rate.
