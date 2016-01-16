import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

class BaseController:
    """Learning agent

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def observe_and_control(self, observation):
        return

    @abstractmethod
    def update_agent(self, reward):
        return

    @abstractmethod
    def reset(self):
        return

class DummyController(BaseController):
    def __init__(self):
        self.control_str_list = ["serve_all", "queue_all", "random_serve_and_queue"]
        self.epoch = None
        self.reset()

    def observe_and_control(self, observation):
        """Dummy is hard-working: it serves all traffic if any, sleep if None.

        :param observation:
        :return:
        """
        traffic_ob, server_ob = observation
        num_req, num_bytes = traffic_ob
        q_len = server_ob
        sleep_flag = True if num_req==0 and q_len==0 else False
        control_req = "queue_all" if sleep_flag else "serve_all"
        return sleep_flag, control_req

    def update_agent(self, reward):
        """Dummy insists it is the best and never updates itself.

        :param reward:
        :return:
        """
        self.epoch += 1
        return

    def reset(self):
        self.epoch = 0
