class SomeAgent:
    def __init__(self):
        pass
    def observe_and_act(self, observation=None, reward=None):
        return None

class Channel:
    def __init__(self):
        pass

    def communicate(self, trx):
        ack = []
        return ack

class Node:
    def __init__(self):
        self.agent = SomeAgent()

    def step(self, last_ack):
        observation = None
        reward = self.evaluate_ack_(last_ack)
        action = self.agent.observe_and_act(observation=observation, reward=reward)
        return self.translate_action_(action)

    def reset(self):
        pass

    def evaluate_ack_(self, ack):
        return 0

    def translate_action_(self, action):
        trx = []
        return trx

class SensorNetwork:
    def __init__(self):
        self.nodes = []  # some agents
        self.channel = Channel()
        self.last_ack = None

    def step(self):
        trx = [node.step(self.last_ack[i_n]) for i_n, node in enumerate(self.nodes)]  # make decisions
        ack = self.channel.communicate(trx)
        self.last_ack = ack