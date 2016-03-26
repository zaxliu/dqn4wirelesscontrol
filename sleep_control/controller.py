import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod


class BaseController(object):
    """Learning agent

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def observe_and_control(self, observation, last_reward):
        return

    @abstractmethod
    def reset(self):
        return


class DummyController(BaseController):
    def __init__(self):
        self.control_str_list = ["serve_all", "queue_all", "random_serve_and_queue"]
        self.epoch = 0

    def observe_and_control(self, observation, last_reward=None):
        """Dummy is hard-working: it serves all traffic if any, sleep if None.

        :param observation:
        :return:
        """
        traffic_ob, server_ob = observation
        num_req, num_bytes = traffic_ob
        q_len = server_ob
        sleep_flag = True if num_req==0 and q_len==0 else False
        control_req = "queue_all" if sleep_flag else "serve_all"
        self.epoch += 1
        return sleep_flag, control_req

    def reset(self):
        self.epoch = 0


class QController(BaseController):
    def __init__(self, agent=None):
        if agent is None:
            raise ValueError("Please pass in agent!")
        else:
            self.agent = agent
        self.epoch = 0

    def observe_and_control(self, observation, last_reward=None):
        if observation is not None:
            (num_req, num_bytes), q_len = observation
            ob_vector = (num_req, q_len)  # total back pressure
        else:
            ob_vector = None
        ((sleep_flag, control_req), _) = self.agent.observe_and_act(observation=ob_vector, reward=last_reward)
        self.epoch += 1
        return sleep_flag, control_req

    def reset(self):
        self.epoch = 0

