from collections import deque

import numpy as np

from hmmlearn.utils import normalize


class PhiMixin(object):
    """Phi function buffers the past PHI_LENGTH (action, observation) pairs
    to form an agent state.
    Currently only support None, 1-d, and 2-d observations and scalar action
    index. The action index is one-hot encoded as a vector to append to the
    last dimension of observation. It is repeated along other dimensions to
    match the shape of observations

    Note: this is a Mixin class, meaning some attributes or methods used
    here is not initialized and need to be implemented somewhere else in
    the inheritance hierarchy.
    """
    def __init__(self, phi_length, **kwargs):
        self.PHI_LENGTH = phi_length
        self.phi_buffer = deque()
        self.__last_state = None  # private attribute
        super(PhiMixin, self).__init__(**kwargs)  # pass on key-word arguments for initialization of parent classes

    def improve_translate_(self, last_observation, last_action, last_reward, observation):
        if last_action is None:  # if no action was taken, return None state
            return None
        idx_action = self.ACTIONS.index(last_action)

        # Update buffer
        state_slice = self.combine_(observation, idx_action)  # get current slice of agent state
        self.phi_buffer.append(state_slice)
        if len(self.phi_buffer) > self.PHI_LENGTH:  # maintain length
            self.phi_buffer.popleft()

        last_state = self.__last_state
        state = np.array(self.phi_buffer) if len(self.phi_buffer)==self.PHI_LENGTH else None
        self.__last_state = state

        # use state as observation and call parent class
        return super(PhiMixin, self).improve_translate_(last_state, last_action, last_reward, state)

    def reset(self, **kwargs):
        self.phi_buffer = deque()
        self.__last_state = None
        super(PhiMixin, self).reset(**kwarg)

        return

    def combine_(self, observation, idx_action):
        """Combine observation and action index into a single tensor

        Parameters
        ----------
        observation : None, 1d, or 2d
        idx_action : scalar
        """
        ob = np.array(observation)
        if len(ob.shape) == 1:
            ac_shape = (len(self.ACTIONS),)
            ac = np.zeros(shape=ac_shape)
            ac[idx_action] = 1
        elif len(ob.shape) == 2:
            ac_shape = (ob.shape[0], len(self.ACTIONS))
            ac = np.zeros(shape=ac_shape)
            ac[:, idx_action] = 1
        else:
            raise ValueError("Unsupported observation shape {}".format(ob.shape))
        state = np.concatenate([ob, ac])

        return state


class AnealMixin(object):
    def __init__(self, recipe=None, **kwargs):
        self.STEPS = []
        self.EPSILONS = []
        self.step_counter = 0
        self.step_ptr = 0
        if recipe is None:
            recipe = {}
        steps = recipe.keys()
        steps.sort()
        for step in steps:
            self.STEPS.append(step)
            self.EPSILONS.append(recipe[step])
        super(AnealMixin, self).__init__(**kwargs)

    def observe_and_act(self, observation, last_reward=None):
        if self.step_ptr < len(self.STEPS):
            if self.step_counter > self.STEPS[self.step_ptr]:
                self.EPSILON = self.EPSILONS[self.step_ptr]
                self.step_ptr += 1
            self.step_counter += 1
        return super(AnealMixin, self).observe_and_act(observation, last_reward)


class LossAnealMixin(object):
    def __init__(self, scale=1, **kwargs):
        self.SCALE = scale
        super(LossAnealMixin, self).__init__(**kwargs)

    def observe_and_act(self, observation, last_reward=None):
        action, update_result = super(LossAnealMixin, self).observe_and_act(observation, last_reward)
        if update_result is not None:
            self.EPSILON = max(min(update_result/self.SCALE, 1), 0)
        return action, update_result


class DynaMixin(object):
    """Model-assisted Reinforcement Learning."""
    def __init__(self, env_model, num_sim=1, verbose=1, **kwargs):
        super(DynaMixin, self).__init__(**kwargs)

        self.env_model = env_model  # environment model
        self.NUM_SIM = num_sim  # number of simulated experience per real experience
        self.__last_state = None
        self.verbose = verbose

    def reset(self, foget_model=False, **kwargs):
        if foget_model:
            self.env_model.reset()
        super(DynaMixin, self).reset(**kwargs)

        return

    def improve_translate_(self, last_observation, last_action, last_reward, observation):
        self.env_model.improve(last_observation, last_action, last_reward, observation)

        last_state = self.__last_state
        state = self.env_model.update_belief_state(last_state, last_action, observation)
        self.__last_state = state

        if self.verbose > 0:
            print " "*4 + "DynaMixin.improve_translate_():",
            print "belief state {}".format(state)

        return super(DynaMixin, self).improve_translate_(last_state, last_action, last_reward, state)

    def reinforce_(self, last_state, last_action, last_reward, state):
        # RL with real experience, presumably implemented by super class
        update_result_real = super(DynaMixin, self).reinforce_(last_state, last_action, last_reward, state)

        # RL with simulated experience
        update_result_sim = [None]*self.NUM_SIM
        for n in range(self.NUM_SIM):
            exp = self.simulate_()
            if self.verbose > 0:
                print " "*4 + "DynaMixin.reinforce_():",
                print "simulated experience {}:".format(n),
                print "({}, {}, {}, {})".format(*exp)
            update_result_sim[n] = super(DynaMixin, self).reinforce_(*exp)

        return (update_result_real, update_result_sim)

    def simulate_(self):
        """Generate simulated experience using environment model
        Procedure:
            1. sample state from state space
            2. perform action (not necessaryly following policy)
            3. collect next state and corresponding reward
        """
        return self.env_model.sample_experience(
            policy=lambda x: self.act_(None)
        )


