from qlearning.qtable import QAgent
from wsn.channel import BaseChannel as Channel
from wsn.sensor_node import BaseNode as Node


class SensorNetwork:
    def __init__(self, nodes=None, channel=None):
        self.nodes = [Node()] if nodes is None else nodes
        self.channel = Channel() if channel is None else channel
        self.last_ack = None

    def step(self):
        trx_and_reward = [node.step(self.last_ack[i_n] if self.last_ack is not None else None) for i_n, node in enumerate(self.nodes)]  # make decisions
        trx = [ele[0] for ele in trx_and_reward]
        rewards = [ele[1] for ele in trx_and_reward]
        ack = self.channel.communicate(trx)
        self.last_ack = ack
        return rewards

    def reset(self):
        map(Node.reset, self.nodes)
        self.channel.reset()
        self.last_ack = None


if __name__=='__main__':
    channel_length=1
    ACTIONS = [(channel_idx, trx_code) for channel_idx in range(channel_length) for trx_code in [-1, 0, 1]]
    agents = [QAgent(actions=ACTIONS, alpha=0.5, gamma=0.5, explore_strategy='epsilon', epsilon=0.1) for i in range(3)]
    nodes = [Node(id=0, agent=agents[0], routing_childs=[None], parent=1, epoch=2, histLen=2, chanLen=channel_length),
            Node(id=1, agent=agents[1], routing_childs=[0], parent=None, epoch=3, histLen=2, chanLen=channel_length)]
            # Node(id=2, agent=agents[2], routing_childs=[0, 1], parent=None, epoch=10, histLen=2)]
    network = SensorNetwork(nodes=nodes, channel=Channel())

    while True:
        rewards = network.step()
        print sum(rewards)


