from channel import BaseChannel
from sensor_node import BaseNode

class SensorNetwork:
    def __init__(self):
        self.nodes = [BaseNode(), BaseNode()]  # some intelligent nodes
        self.channel = BaseChannel()
        self.last_ack = None

    def step(self):
        trx = [node.step(self.last_ack[i_n]) for i_n, node in enumerate(self.nodes)]  # make decisions
        ack = self.channel.communicate(trx)
        self.last_ack = ack