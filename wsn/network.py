import channel.BaseChannel as Channel
import sensor_node.BaseNode as Node


class SensorNetwork:
    def __init__(self, nodes=None, channel=None):
        self.nodes = [Node()] if nodes is None else nodes
        self.channel = Channel() if channel is None else channel
        self.last_ack = None

    def step(self):
        trx = [node.step(self.last_ack[i_n]) for i_n, node in enumerate(self.nodes)]  # make decisions
        ack = self.channel.communicate(trx)
        self.last_ack = ack

    def reset(self):
        map(Node.reset, self.nodes)
        self.channel.reset()
        self.last_ack = None


if __name__=='__main__':
    pass