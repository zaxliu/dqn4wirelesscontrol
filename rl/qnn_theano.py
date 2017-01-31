from collections import deque

import numpy as np
from scipy.stats import itemfreq

import theano
from theano import shared
import theano.tensor as T
import lasagne

from qtable import QAgent
from simple_envs import SimpleMaze


class QAgentNN(QAgent):
    """ Neural-network-based Q Learning Agent
    This agent replaces the Q table in a canonical q agent with a neural net.
    Its inputs are the observed state and its outputs are the q values for
    each action.

    The training of the network is performed periodically with randomly
    selected batch of past experiences. This technique is also known as
    Experience Replay. The loss function is defined following the Bellman
    iteration equation.

    Different from the techniques presented in the original DeepMind paper.
    We apply re-scaling on the reward to make it fit better into the value
    range of output layer (e.g. (-1, +1)). This can be helpful for scenarios
    in which the value of reward has a large dynamic range. Another
    modification is that we employ separate buffers in the replay memory
    for different actions. This can speed-up convergence in non-stationary
    and highly-action-skewed cases.

    QAgentNN reuses much of the interface methods of QAgent. The reset(),
    imporove_translate_(), reinforce_(), update_table_(), lookup_table_(),
    and act_() methods are redefined to overwrite/encapsulate the original
    functionality.
    """
    def __init__(self,
                 dim_state, range_state,                            # state
                 f_build_net=None,                                  # network
                 batch_size=100, learning_rate=0.01, momentum=0.9,  # SGD
                 update_period=1, freeze_period=1,                  # schedule
                 reward_scaling=1, reward_scaling_update='fixed',  rs_period=1, # reward
                 memory_size=500, num_buffer=1,                     # memory
                 **kwargs):
        """Initialize NN-based Q Agent

        Parameters
        ----------
        dim_state   : dimensions of observation. 3-D tensor.
        range_state : lower and upper bound of each observation dimension.
            4-D tensor (d1, d2, d3, 2).
        f_build_net : Function handle for building DQN, and returns a
            Lasagne output layer.
        batch_size  : batch size for mini-batch SGD.
        learning_rate  : step size of a single gradient descent step.
        momentum    : faction of old weight values kept during gradient descent.
        reward_scaling : initial value for inverse reward scaling factor
        reward_scaling_update : 'fixed' & 'adaptive'
        update_period  : number of epochs per SGD update.
        freeze_period  : number of SGD updates per target network sync.
        memory_size : size of replay memory (each buffer).
        num_buffer  : number of buffers used in replay memory
        kwargs      :

        Returns
        -------

        """
        super(QAgentNN, self).__init__(**kwargs)

        self.DIM_STATE = dim_state  # mush be in form (d1, d2, d3), i.e. three dimensions
        if range_state is not None:  # lower and upper bound on observation
            self.STATE_MEAN = (np.array(range_state)[:, :, :, 1]+np.array(range_state)[:, :, :, 0])/2.0
            self.STATE_MAG = (np.array(range_state)[:, :, :, 1]-np.array(range_state)[:, :, :, 0])/2.0
        else:
            self.STATE_MEAN = np.zeros(self.DIM_STATE)
            self.STATE_MAG = np.ones(self.DIM_STATE)

        self.FREEZE_PERIOD = freeze_period
        self.MEMORY_SIZE = memory_size
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.MOMENTUM = momentum
        self.UPDATE_PERIOD = update_period
        self.REWARD_SCALING_UPDATE = reward_scaling_update
        self.RS_PERIOD = rs_period
        self.fun_build_net = f_build_net if f_build_net is not None else QAgentNN.build_net_

        self.rs = reward_scaling
        self.freeze_counter = self.FREEZE_PERIOD - 1
        self.update_counter = self.UPDATE_PERIOD - 1
        self.rs_counter = self.RS_PERIOD - 1

        # set q table as a NN
        self.fun_train_qnn, self.fun_adapt_rs, self.fun_clone_target, self.fun_q_lookup, self.fun_rs_lookup = \
            self.init_fun_(self.fun_build_net,
                           self.DIM_STATE, self.BATCH_SIZE, self.GAMMA,
                           self.LEARNING_RATE, self.MOMENTUM,
                           self.rs, self.REWARD_SCALING_UPDATE)

        self.replay_memory = QAgentNN.ReplayMemory(memory_size, batch_size, dim_state, len(self.ACTIONS), num_buffer)

    def reset(self, foget_table=False, foget_memory=False, **kwargs):
        self.freeze_counter = self.FREEZE_PERIOD - 1
        self.update_counter = self.UPDATE_PERIOD - 1
        self.rs_counter = self.RS_PERIOD - 1

        if foget_table:
            self.fun_train_qnn, \
            self.fun_adapt_rs, \
            self.fun_clone_target, \
            self.fun_q_lookup, \
            self.fun_rs_lookup = self.init_fun_(
                self.fun_build_net,
                self.DIM_STATE, self.BATCH_SIZE, self.GAMMA, self.LEARNING_RATE, self.MOMENTUM,
                self.rs, self.REWARD_SCALING_UPDATE
            )

        if foget_memory:
            self.ReplayMemory.reset()

        kwargs['foget_table'] = foget_table
        super(QAgentNN, self).reset(**kwargs)

        return

    def improve_translate_(self, last_observation, last_action, last_reward, observation):
        # try invoking super-class
        return super(QAgentNN, self).improve_translate_(
            last_observation, last_action, last_reward, observation
        )

    def reinforce_(self, last_state, last_action, last_reward, state):
        """Train the network
        Put the latest experience into replay memory and eriodically train
        the network with batch of random samples sampled from the replay
        memory.

        Freeze parameters of the target network during non-training epochs.

        Will not update the network if the state or reward passes in is None
        or the replay memory is yet to be filled up.

        Parameters
        ----------
        last_state  : last agent state
        last_action :
        last_reward : reward from last action
        state       :

        Returns     : training loss or None
        -------

        """
        # Check if this is a valid experience
        if last_state is None:
            if self.verbose > 0:
                print " "*4 + "QAgentNN.reinforce_():",
                print "last_state is None."
            return None
        elif last_reward is None:
            if self.verbose > 0:
                print " "*4 + "QAgentNN.reinforce_():",
                print "last_reward is None."
            return None
        elif state is None:
            if self.verbose > 0:
                print " "*4 + "QAgentNN.reinforce_():",
                print "state is None."
            return None

        # Store latest experience into replay memory
        idx_action = self.ACTIONS.index(last_action)
        self.replay_memory.update(last_state, idx_action, last_reward, state)

        # Update networks
        if not self.replay_memory.isfilled():
            if self.verbose > 0:
                print " "*4 + "QAgentNN.reinforce_():",
                print "unfull memory."
        else:
            # Log counter progress 
            if self.verbose > 0:
                print " "*4 + "QAgentNN.reinforce_():",
                print "update counter {}, freeze counter {}, rs counter {}.".format(
                    self.update_counter, self.freeze_counter, self.rs_counter)

            loss = None
            if self.update_counter == 0:
                last_states, last_actions, last_rewards, states = self.replay_memory.sample_batch()
                loss = self.update_table_(last_states, last_actions, last_rewards, states)
                self.update_counter = self.UPDATE_PERIOD - 1

                if self.verbose > 0:
                    print " "*4 + "QAgentNN.reinforce_():",
                    print "update loss is {}, reward_scaling is {}".format(
                        loss, self.rs
                    )

                if self.verbose > 1:
                    freq = itemfreq(last_actions)
                    print " "*8 + "QAgentNN.reinforce_():",
                    print "batch action distribution: {}".format(
                        {self.ACTIONS[int(freq[i, 0])]: 1.0 * freq[i, 1] /
                         self.BATCH_SIZE for i in range(freq.shape[0])}
                    )
            else:
                self.update_counter -= 1

            return loss

        return None

    def act_(self, state):
        # Escalate to QAgent.act_().
        # Pass None state if memory is not full to invoke random action.
        if self.is_memory_filled():
            return super(QAgentNN, self).act_(state)
        else:
            if self.verbose > 2:
                print " "*8 + "QAgentNN.act_():",
                print "unfull memory."
            return super(QAgentNN, self).act_(None)

    def update_table_(self, last_state, last_action, reward, current_state):
        loss = self.fun_train_qnn(
            self.rescale_state(last_state),
            last_action,
            reward,
            self.rescale_state(current_state)
        )

        # Sync target network with non-target network
        if self.freeze_counter == 0:
            self.fun_clone_target()
            self.freeze_counter = self.FREEZE_PERIOD - 1
        else:
            self.freeze_counter -= 1

        # adjust reward scaling values
        if self.REWARD_SCALING_UPDATE=='adaptive':
            if self.rs_counter == 0:
                loss = self.fun_adapt_rs(
                    self.rescale_state(last_state),
                    last_action,
                    reward,
                    self.rescale_state(current_state)
                )
                self.rs_counter = self.RS_PERIOD - 1
                self.rs = self.fun_rs_lookup()
            else:
                self.rs_counter -= 1

        return loss

    def lookup_table_(self, state):
        state_var = np.zeros(tuple([1]+list(self.DIM_STATE)), dtype=np.float32)
        state_var[0, :] = state
        return self.fun_q_lookup(self.rescale_state(state_var)).ravel().tolist()

    def is_memory_filled(self):
        return self.replay_memory.isfilled()

    def rescale_state(self, states):
        return (states-self.STATE_MEAN)/self.STATE_MAG

    @staticmethod
    def build_net_(input_var=None, input_shape=None, num_outputs=None):
        if input_shape is None or num_outputs is None:
            raise ValueError('State or Action dimension not given!')
        l_in = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
        l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=500,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units=500,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        l_out = lasagne.layers.DenseLayer(
            l_hid2, num_units=num_outputs,
            nonlinearity=lasagne.nonlinearities.tanh)
        return l_out

    def init_fun_(self, f_build_net,
                  dim_state, batch_size, gamma,
                  learning_rate, momentum,
                  reward_scaling, reward_scaling_update):
        """Define and compile function to train and evaluate network
        :param f_build_net: function to build dqn
        :param dim_state: dimensions of a single state tensor
        :param batch_size:
        :param gamma: future reward discount factor
        :param learning_rate:
        :param momentum:
        :param reward_scaling:
        :param reward_scaling_update:
        :return:
        """
        self.qnn = f_build_net(
            None, tuple([None]+list(self.DIM_STATE)), len(self.ACTIONS)
        )
        self.qnn_target = f_build_net(
            None, tuple([None]+list(self.DIM_STATE)), len(self.ACTIONS)
        )

        if len(dim_state) != 3:
            raise ValueError("We only support 3 dimensional states.")

        # inputs
        # state: (BATCH_SIZE, MEMORY_LENGTH, DIM_STATE[0], DIM_STATE[1])
        old_states, new_states = T.tensor4s('old_states', 'new_states')
        actions = T.ivector('actions')           # (BATCH_SIZE, 1)
        rewards = T.vector('rewards')            # (BATCH_SIZE, 1)
        rs = shared(value=reward_scaling*1.0, name='reward_scaling')

        # intermediates
        predict_q = lasagne.layers.get_output(
            layer_or_layers=self.qnn, inputs=old_states
        )
        predict_q_inference = lasagne.layers.get_output(
            layer_or_layers=self.qnn, inputs=old_states, deterministic=True
        )
        predict_next_q = lasagne.layers.get_output(
            layer_or_layers=self.qnn_target, inputs=new_states
        )
        target_q = rewards/rs + gamma*T.max(predict_next_q, axis=1)

        # penalty
        singularity = 1+1e-4
        penalty = T.mean(
            1/T.pow(predict_q[T.arange(batch_size), actions]-singularity, 2) +
            1/T.pow(predict_q[T.arange(batch_size), actions]+singularity, 2) -
            2
        )


        # outputs
        loss = T.mean(
            (predict_q[T.arange(batch_size), actions] - target_q)**2
        ) + (1e-5)*penalty

        # weight update formulas (mini-batch SGD with momentum)
        params = lasagne.layers.get_all_params(self.qnn, trainable=True)
        grads = T.grad(loss, params)
        grads = lasagne.updates.total_norm_constraint(grads, 10)
        updates = lasagne.updates.nesterov_momentum(
            grads, params, learning_rate=learning_rate, momentum=momentum
        )
        updates_rs = lasagne.updates.nesterov_momentum(
            loss, [rs], learning_rate=learning_rate, momentum=momentum
        )

        # functions
        fun_train_qnn = theano.function(
            [old_states, actions, rewards, new_states],
            loss, updates=updates, allow_input_downcast=True
        )
        fun_adapt_rs = theano.function(
            [old_states, actions, rewards, new_states],
            loss, updates=updates_rs, allow_input_downcast=True
        )

        def fun_clone_target():
            lasagne.layers.helper.set_all_param_values(
                self.qnn_target,
                lasagne.layers.helper.get_all_param_values(self.qnn)
            )

        fun_q_lookup = theano.function(
            [old_states], predict_q_inference, allow_input_downcast=True
        )
        fun_rs_lookup = rs.get_value

        return fun_train_qnn, fun_adapt_rs, fun_clone_target, fun_q_lookup, fun_rs_lookup

    class ReplayMemory(object):
        """Replay memory
        Buffers the past "memory_size) (s, a, r, s') tuples in a circular buffer, and provides method to sample a random
        batch from it.
        """
        def __init__(self, memory_size, batch_size, dim_state, num_actions, num_buffers=1):
            self.MEMORY_SIZE = memory_size
            self.BATCH_SIZE = batch_size
            self.DIM_STATE = dim_state
            self.NUM_ACTIONS = num_actions
            self.NUM_BUFFERS = num_buffers

            self.buffer_old_state = np.zeros(tuple([self.NUM_BUFFERS, memory_size]+list(self.DIM_STATE)), dtype=np.float32)
            self.buffer_action = np.zeros((self.NUM_BUFFERS, memory_size, ), dtype=np.int32)
            self.buffer_reward = np.zeros((self.NUM_BUFFERS, memory_size, ), dtype=np.float32)
            self.buffer_new_state = np.zeros(tuple([self.NUM_BUFFERS, memory_size]+list(self.DIM_STATE)), dtype=np.float32)

            self.top = [-1]*self.NUM_BUFFERS
            self.filled = [False]*self.NUM_BUFFERS

        def update(self, last_state, idx_action, last_reward, new_state):
            buffer_idx = idx_action % self.NUM_BUFFERS
            top = (self.top[buffer_idx]+1) % self.MEMORY_SIZE
            self.buffer_old_state[buffer_idx, top, :] = last_state
            self.buffer_action[buffer_idx, top] = idx_action
            self.buffer_reward[buffer_idx, top] = last_reward
            self.buffer_new_state[buffer_idx, top, :] = new_state
            if not self.filled[buffer_idx]:
                self.filled[buffer_idx] |= (top == (self.MEMORY_SIZE-1))
            self.top[buffer_idx] = top

        def sample_batch(self):
            sample_idx = np.random.randint(0, self.MEMORY_SIZE, (self.BATCH_SIZE,))
            buffer_idx = np.random.randint(0, self.NUM_BUFFERS, (self.BATCH_SIZE,))
            return (self.buffer_old_state[buffer_idx, sample_idx, :],
                    self.buffer_action[buffer_idx, sample_idx],
                    self.buffer_reward[buffer_idx, sample_idx],
                    self.buffer_new_state[buffer_idx, sample_idx, :])

        def isfilled(self):
            return all(self.filled)

        def reset(self):
            self.top = [-1]*self.NUM_BUFFERS
            self.filled = [False]*self.NUM_BUFFERS

