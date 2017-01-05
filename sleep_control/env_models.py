##################################################
# Environment Model for SJTU trace + BS sleeping
# Author: Jingchu Liu
##################################################

import sys
sys.path.append('../')
from collections import deque

import numpy as np
from numpy.random import randint

from scipy.stats import poisson

from rl.env_models import BaseEnvModel
from rl.mmpp import MMPP

from hmmlearn.utils import normalize


class SJTUModel(BaseEnvModel):
    """Environment Model for BS sleeping in SJTU traffic trace.

    Observation:
        (last queue length, last request arrival, current queue length).
        Cannot know traffic arrival in current epoch in advance. [But last queue length redundant?]
    State: Tuple.
        Elements by position:
            last traffic belief state, belief=wake probability. [0, 1]
            current queue length. [0, \infty)
            last sleeping state. {0, 1})
        Quantization:
            Support raw or quantized traffic belief state, quantization is
            linear. Maximum bins for each state is given as parameter.
    Model:
        Transitions:
            Traffic follow MMPP or IPP, independent of actions [approximation, or assume two models?]
            last_q + last_traffic - last_served = current_q, current_q = 0 if served in the last epoch.
            sleep state straightforwardly follows sleeping control actions.
        Reward:
            Deterministic part:
                Service reward, delay penalty, energy penalty.
            Stochastic part:
                Time-out. Assume rare and do not model.
        Sensery:
            Emission model is Poisson.
    """
    def __init__(self, traffic_params, queue_params, reward_params, **kwargs):
        """
        Parameters
        ----------
        traffic_params : traffic model parameters. Tuple, entries for each position:
            model_type : type of traffic model 'Poisson', 'IPP', or 'MMPP'.
            traffic_window_size : size of moving window of traffic observations.
            adjust_offset : small offset to avoid over-/underflow during model fitting.
            n_belief_bins : number of bins to quantize wake probability. Return raw state if 0.
        queue_params : queue related parameters.
            max_queue_len : maximum queue length to show in state.
        reward_params : parameters related to determinsitic rewards.
            serve  : reward for serving each request
            wait   : cost for queueing a request for one epoch
            fail   : cost for rejecting or time-out each request
            energy : cost for waking up server for one epoch
        """
        (model_type, traffic_window_size, adjust_offset, n_belief_bins) = traffic_params
        (max_queue_len, ) = queue_params
        (Rs, Rw, Rf, Re) = reward_params

        # static params
        self.TRAFFIC_MODEL_TYPE = model_type
        self.TRAFFIC_WINDOW_SIZE = traffic_window_size
        self.ADJUST_OFFSET = adjust_offset
        self.N_BELIEF_BINS = n_belief_bins
        delta = 1.0 / self.N_BELIEF_BINS if self.N_BELIEF_BINS != 0 else None
        self.BELIEF_BIN_VALS = np.arange(0, 1+delta, delta) if delta is not None else None
        self.MAX_Q_LEN = max_queue_len
        self.R_SERVE = Rs
        self.R_WAIT = Rw
        self.R_FAIL = Rf
        self.R_ENERGY = Re

        # dynamic attr.
        self.traffic_model = None
        self.traffic_window = deque()

        # procedures
        self.init_traffic_model_()

    def improve(self, last_observation, last_action, last_reward, observation):
        """Improve environment model.
        The only trainable part of env model is the traffic model. Use a window
        of past observations to fit the traffic model.
        """
        # TODO: fit stride and n_iteration
        (last_q, last_traffic_req, current_q) = observation

        # Update traffic moving window and fit model
        if last_traffic_req is not None:
            self.traffic_window.append(last_traffic_req)

        if len(self.traffic_window) > self.TRAFFIC_WINDOW_SIZE:
            self.traffic_window.popleft()
        else:
            pass  # TODO: vervose information.

        # Re-fit traffic model
        if len(self.traffic_window) == self.TRAFFIC_WINDOW_SIZE:
            self.traffic_model.fit(self.traffic_window)
            self.adjust_traffic_model_()

        return

    def score(self, n=100):
        """Per-step log likelihood of real and simulated observation (appox. expectation)"""
        score_window = self.traffic_model.score(self.traffic_window) / self.TRAFFIC_WINDOW_SIZE
        score_expected = self.traffic_model.score(self.traffic_model.sample(n)[0]) / n

        return (score_window, score_expected)

    def update_belief_state(self, last_state, last_action, observation):
        """Estimate belief state using current observation.

        belief_state = (last_traffic_belief, current_q, last_sleep_flag)
        """
        if observation is None or len(self.traffic_window) < self.TRAFFIC_WINDOW_SIZE:
            return None

        (last_q, last_traffic_req, current_q) = observation
        (sleep_flag, control_req) = last_action

        # Get traffic belief state. Assume traffic window already contain latest observation
        framelogprob = self.traffic_model._compute_log_likelihood(
            np.array(self.traffic_window)[:, None])  # observation log prob. for each time step
        _, fwdlattice = self.traffic_model._do_forward_pass(
            framelogprob)  # log posteriors for each time step
        posterior = np.exp(fwdlattice[-1, :]); normalize(posterior)
        traffic_belief = self.quantize_belief_state_(posterior[0])

        # current queue = min(current queue, MAX_Q_LEN)
        current_q = self.limit_queue_length(current_q)

        # last sleeping state = last sleep flag
        pass

        return (traffic_belief, current_q, sleep_flag)

    def sample_experience(self, **kwargs):
        state, action, reward, next_state = \
                super(SJTUModel, self).sample_experience(**kwargs)
        state = (self.quantize_belief_state_(state[0]),
                 self.limit_queue_length(state[1]),
                 state[2])
        next_state = (self.quantize_belief_state_(next_state[0]),
                      self.limit_queue_length(next_state[1]),
                      next_state[2])

        return state, action, reward, next_state

    def sample_state_(self):
        """Sample state from state space."""
        traffic_belief = np.random.rand()  # uniform random [0, 1]
        current_q = np.random.randint(
            0, self.MAX_Q_LEN+1
        )  # uniform random [0, max_q]
        last_sleep_flag = np.random.rand() < 0.5  # 50-50 sleep
        return traffic_belief, current_q, last_sleep_flag

    def sample_transition_(self, state, action):
        """Sample next state and reward conditioned on state and action."""
        (last_traffic_belief, current_q, last_sleep_flag) = state
        (sleep_flag, control_req) = action

        # Traffic state
        # predict prior belief for current traffic state
        cur_traffic_pred = np.matmul(
            np.array([last_traffic_belief, 1-last_traffic_belief]),
            self.traffic_model.transmat_)
        normalize(cur_traffic_pred)
        # sample current traffic state, 0: wake, 1: sleep
        cur_traffic_state = int(np.random.rand() < cur_traffic_pred[0])
        # sample current observation 
        cur_traffic_ob = poisson.rvs(
            self.traffic_model.emissionrates_[cur_traffic_state]
        )
        # compute posterior belief for current traffic state
        posterior = poisson.pmf(
            cur_traffic_ob, self.traffic_model.emissionrates_) * \
            cur_traffic_pred
        normalize(posterior)

        # Queue state
        total_req = current_q + cur_traffic_ob
        next_q = total_req if sleep_flag==True else 0  # queue all or serve all


        # Reward
        reward = self.R_SERVE * total_req * int(not sleep_flag) + \
                 self.R_WAIT * total_req * int(sleep_flag) + \
                 self.R_ENERGY * int(not sleep_flag)

        return (posterior[0], next_q, sleep_flag), reward

    def init_traffic_model_(self):
        """Initialize HMMLearn traffic model."""
        if self.TRAFFIC_MODEL_TYPE == 'Poisson':
            self.traffic_model = MMPP(
                n_components=1, n_iter=1, init_params='', verbose=False)
            self.traffic_model.startprob_ = np.array([1.0])
            self.traffic_model.transmat_ = np.array([1.0])
            self.traffic_model.emissionrates_ = np.array([1.0])
        elif self.TRAFFIC_MODEL_TYPE == 'IPP':
            self.traffic_model = MMPP(
                n_components=2, n_iter=1, init_params='', verbose=False)
            self.traffic_model.startprob_ = np.array([.5, .5])
            self.traffic_model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
            self.traffic_model.emissionrates_ = np.array([1.0, 0.0])
        elif self.TRAFFIC_MODEL_TYPE == 'MMPP':
            self.traffic_model = MMPP(
                n_components=2, n_iter=1, init_params='', verbose=False)
            self.traffic_model.startprob_ = np.array([.5, .5])
            self.traffic_model.transmat_ = np.array([[0.6, 0.4], [0.4, 0.6]])
            self.traffic_model.emissionrates_ = np.array([0.9, 1.1])
        else:
            raise ValueError(
                'Unknown traffic model type{}'.format(self.MODEL_TYPE)
            )

        return

    def adjust_traffic_model_(self):
        """Offset model params to avoid over-/under-flow"""
        # TODO: replace NaN and show warning.
        self.traffic_model.startprob_ = \
            np.nan_to_num(self.traffic_model.startprob_)
        self.traffic_model.transmat_ = \
            np.nan_to_num(self.traffic_model.transmat_)
        self.traffic_model.emissionrates_ = \
            np.nan_to_num(self.traffic_model.emissionrates_)

        self.traffic_model.startprob_prior += self.ADJUST_OFFSET
        self.traffic_model.transmat_ += self.ADJUST_OFFSET
        if self.TRAFFIC_MODEL_TYPE == 'Poisson' or 'MMPP':
            self.traffic_model.emissionrates_ += self.ADJUST_OFFSET  # when the model is general MMPP
        elif self.TRAFFIC_MODEL_TYPE == 'IPP':
            self.traffic_model.emissionrates_[0] += self.ADJUST_OFFSET
            self.traffic_model.emissionrates_[1] = 0.0  # when the model is IPP
        else:
            raise ValueError('Unknown traffic model type {}'.format(self.MODEL_TYPE))


        normalize(self.traffic_model.startprob_)
        normalize(self.traffic_model.transmat_, axis=1)

        return

    def quantize_belief_state_(self, belief):
        """Quantize traffic belief and return Bin Values."""
        if self.N_BELIEF_BINS > 0:
            minidx = np.argmin(np.abs(self.BELIEF_BIN_VALS - belief))
            return self.BELIEF_BIN_VALS[minidx]
        else:
            return belief

    def limit_queue_length(self, q_len):
        """Limit the maximum queue length in state."""
        return self.MAX_Q_LEN if q_len > self.MAX_Q_LEN else q_len


