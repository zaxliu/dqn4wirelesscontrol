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
            last_q_len, last_traffic_req, new_q_len = observation
            ob_vector = (last_q_len, last_traffic_req, new_q_len)  # total back pressure
        else:
            ob_vector = None
        (sleep_flag, control_req), update_result = self.agent.observe_and_act(observation=ob_vector, last_reward=last_reward)
        self.epoch += 1
        return (sleep_flag, control_req), update_result

    def reset(self):
        self.epoch = 0


class NController(BaseController):
    def __init__(self, N_on=10, N_off=0):
        self.N_on = N_on
        self.N_off = N_off
        self.last_action = None
        self.epoch = 0

    def observe_and_control(self, observation, last_reward=None):
        if observation is not None:
            last_q_len, last_traffic_req, new_q_len = observation
        else:
            last_q_len, last_traffic_req, new_q_len = 0, 0, 0
        if new_q_len >= self.N_on:
            (sleep_flag, control_req) = (False, 'serve_all')
        elif new_q_len <= self.N_off:
            (sleep_flag, control_req) = (True, None)
        else:
            (sleep_flag, control_req) = self.last_action if self.last_action is not None else (True, None)
        self.last_action = (sleep_flag, control_req)
        self.epoch += 1
        return (sleep_flag, control_req), None

    def reset(self):
        self.epoch = 0
