class SomeAgent:
    def __init__(self):
        pass
    def observe_and_act(self, observation=None, reward=None):
        return None


class BaseNode(object):
    def __init__(self):
        self.agent = SomeAgent()  # some intelligent agent with a learning and decision interface
        self.traffic = None  # responsible for generating traffic
        self.routing_childs = None  # who to receive
        self.epoch = 0  # time counter

    def step(self, last_ack):
        observation = None  # past actions + traffic info + etc...
        reward = self.evaluate_ack_(last_ack)
        action = self.agent.observe_and_act(observation=observation, reward=reward)
        return self.translate_action_(action)

    def reset(self):
        pass

    def evaluate_ack_(self, ack):
        """Evaluate last TRX and emit reward
        """
        return 0

    def translate_action_(self, action):
        # translate action in to trx cmd
        trx = []
        return trx