from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.random import randint


class BaseEnvModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        pass

    def reset(self, **kwargs):
        pass

    @abstractmethod
    def improve(self, last_observation, last_action, last_reward, observation):
        """Use new experience to improve model."""
        pass

    @abstractmethod
    def update_belief_state(self, last_state, last_action, observation):
        """Update belief state based on current experience. a.k.a filtering."""
        pass

    def sample_experience(self, policy, **kwargs):
        """Generate simulated experience using environment model
        Procedure:
            1. sample state from state space
            2. perform action (not necessaryly following policy)
            3. collect next state and corresponding reward
        """
        # randomly sample state
        state = self.sample_state_()
        # randomly act in sampled state
        action = policy(state)
        # generate reward and next state
        next_state, reward = self.sample_transition_(state, action)

        return state, action, reward, next_state

    @abstractmethod
    def sample_state_(self, **kwargs):
        """Randomly sample a state from state space"""
        pass

    @abstractmethod
    def sample_transition_(self, state, action):
        """Randomly sample next state from transition P(s'|s, a) and reward
        from reward function R(r|s, a, [s'])"""
        pass


class SimpleMazeModel(BaseEnvModel):
    def __init__(self, maze, **kwargs):
        self.maze = maze

    def reset(self, **kwargs):
        pass

    def improve(self, last_observation, last_action, last_reward, observation):
        # I have no model to improve
        pass

    def update_belief_state(self, last_state, last_action, observation):
        # SimpleMaze if fully observable
        return observation

    def sample_state_(self, **kwargs):
        return(randint(0, self.maze.DIMS[0]), randint(0, self.maze.DIMS[1]))

    def sample_transition_(self, state, action):
        return self.maze.transition_(state, action)

