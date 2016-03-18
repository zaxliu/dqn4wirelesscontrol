from traffic_emulator import TrafficEmulator
from traffic_server import TrafficServer
from controller import DummyController

class Environment:
    """Glue logic

    """
    def __init__(self, te=None, ts=None):
        self.te = TrafficEmulator() if te is None else te
        self.ts = TrafficServer if ts is None else ts
        self.epoch = None
        self.reset()


    def get_observation(self):
        traffic_df = self.te.generate_traffic()
        if traffic_df is None:
            print "Run out of data, please reset environment!"
            return None
        else:
            self.ts.feed_traffic(traffic_df=traffic_df)
            observation = self.ts.observe()
            return observation

    def control_and_reward(self, control):
        service_df, cost = self.ts.get_service_and_cost(control=control)
        reward = self.te.serve_and_reward(service_df=service_df)
        self.epoch += 1
        return reward-cost

    def reset(self):
        self.te.reset()
        self.ts.reset()
        self.epoch = 0

class Emulation:
    def __init__(self, te=None, ts=None, c=None):
        te = TrafficEmulator() if te is None else te
        ts = TrafficServer() if ts is None else ts
        c = DummyController() if c is None else c
        self.e = Environment(te=te, ts=ts)
        self.c = c
        self.epoch = None
        self.reset()

    def step(self):
        observation = self.e.get_observation()
        if observation is None:
            print "Run out of data, please reset!"
            return None
        control = self.c.observe_and_control(observation=observation)
        reward = self.e.control_and_reward(control=control)
        self.c.update_agent(reward)
        self.epoch += 1
        return self.epoch-1, observation, control, reward

    def reset(self):
        self.e.reset()
        self.c.reset()
        self.epoch = 0

