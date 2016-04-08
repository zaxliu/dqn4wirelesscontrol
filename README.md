# DQN for Wireless Control
This projects contains a python-based implementation of Q learning agent as well as two example applications in wireless network control.

We implemented both the canonnical table-based agent in qlearning.qtable.QAgent and the more recent neuron-network-based version, a.k.a DQN, in qlearning.qnn.QAgentNN. And they share the same programming interface which can be handy when one want to start with a simple interpretable agent and then evolve to a more sophisticated one.

For non-Markovian environments, we also provide a mixin class in qlearning.mixin.PhiMixin. It can be used with either type of agents to stack the historical observation into a augmented observation vector. For better exploration and exploitation tradeoff, we implemented a anealing mixin class in qlearning.mixin.AnealMixin for gradually decrease exploration rate.
