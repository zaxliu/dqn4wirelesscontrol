from traffic_emulator import TrafficEmulator
from traffic_server import TrafficServer
from controller import DummyController


class Emulation:
    def __init__(self, te=None, ts=None, c=None, beta=None):
        self.te = TrafficEmulator() if te is None else te
        self.ts = TrafficServer() if ts is None else ts
        self.c = DummyController() if c is None else c
        self.epoch = 0
        self.last_reward = None
        self.last_cost = None
        self.BETA = beta

    def step(self):
        if self.last_reward is None or self.last_cost is None:
            system_reward = None
        else:
            system_reward = (self.last_reward + self.last_cost) if self.BETA is None \
                else (self.BETA*self.last_reward + (1-self.BETA)*self.last_cost)
        print "Emulation.step():",
        print "last reward: {}".format(system_reward)

        observation = self.get_observation_()
        print "Emulation.step():",
        print "observation: {}".format(observation)
        if observation is None:
            print "Emulation.step():",
            print "run out of data, please reset!"
            return None

        control, update_result = self.c.observe_and_control(observation=observation, last_reward=system_reward)
        print "Emulation.step():",
        print "control: {}, agent update: {}".format(control, update_result)

        cost, reward = self.control_and_reward_(control=control)
        print "Emulation.step():",
        print "cost: {}, reward: {}".format(cost, reward)
        self.last_cost = cost
        self.last_reward = reward
        self.epoch += 1
        return observation, control, cost, reward

    def reset(self):
        self.te.reset()
        self.ts.reset()
        self.c.reset()
        self.epoch = 0
        self.last_reward = None
        self.last_cost = None

    def get_observation_(self):
        traffic_df = self.te.generate_traffic()
        if traffic_df is None:
            print "Emulation.step():",
            print "run out of data, please reset environment!"
            return None
        else:
            observation = self.ts.observe(traffic_df=traffic_df)
            return observation

    def control_and_reward_(self, control):
        service_df, cost = self.ts.get_service_and_cost(control=control)
        reward = self.te.serve_and_reward(service_df=service_df)
        return cost, reward
