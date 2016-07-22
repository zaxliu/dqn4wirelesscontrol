from collections import deque

import numpy as np
from scipy.stats import itemfreq

import theano
from theano import shared
import theano.tensor as T
import lasagne

from qtable import QAgent
from qlearning.simple_envs import SimpleMaze


class QAgentNN(QAgent):
    """ Neuron-network-based Q Learning Agent
    This agent replaces the Q table in a canonical q agent with a neuron network. Its inputs are the observed state and
    its outputs are the q values for each action. The training of the network is performed periodically with randomly
    selected batch of past experiences. This technique is also known as Experience Replay. The loss function is defined
    following the Bellman iteration equation.

    Different from the techniques presented in the original DeepMind paper. We apply re-scaling on the reward to make it
    fit better into the value range of output layer (e.g. (-1, +1)). This can be helpful for scenarios in which the value
    of reward has a large dynamic range. Another modification is that we employ separate buffers in the replay memory
    for different actions. This can speed-up convergence in non-stationary and highly-action-skewed cases.

    QAgentNN reuses much of the interface methods of QAgent. The reset(), reinforce_(), transition_(), update_table_(),
    lookup_table_(), and act_() methods are redefined to overwrite/encapsulate the original functionality.
    """
    def __init__(self, dim_state, range_state,  # basics
                 f_build_net=None, freeze_period=1,  # network
                 batch_size=100, learning_rate=0.01, momentum=0.9,  update_period=1,  # training
                 reward_scaling=1, reward_scaling_update='fixed',  rs_period=1, # reward
                 memory_size=500, num_buffer=1,  # memory
                 **kwargs):
        """Initialize NN-based Q Agent

        Parameters
        ----------
        dim_state   : dimensions of observation. Must by in the format of (d1, d2, d3).
        range_state : lower and upper bound of observations. Must be in the format of (d1, d2, d3, 2) following the notation of dim state.
        net         : Lasagne output layer as the network used.
        batch_size  : batch size for mini-batch stochastic gradient descent (SGD) training.
        learning_rate  : step size of a single gradient descent step.
        momentum    : faction of old weight values kept during gradient descent.
        reward_scaling : initial value for inverse reward scaling factor
        reward_scaling_update : 'fixed' & 'adaptive'
        freeze_period  : the periodicity for training the q network.
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
        self.REWARD_SCALING = reward_scaling
        self.REWARD_SCALING_UPDATE = reward_scaling_update
        self.RS_PERIOD = rs_period
        # set q table as a NN
        if not f_build_net:
            f_build_net = QAgentNN.build_net_
        self.qnn = f_build_net(None, tuple([None]+list(self.DIM_STATE)), len(self.ACTIONS))
        self.qnn_target = f_build_net(None, tuple([None]+list(self.DIM_STATE)), len(self.ACTIONS))
        self.fun_train_qnn, self.fun_adapt_rs, self.fun_clone_target, self.fun_q_lookup, self.fun_rs_lookup = \
            self.init_fun_(
                dim_state=self.DIM_STATE, batch_size=self.BATCH_SIZE, gamma=self.GAMMA, learning_rate=self.LEARNING_RATE,
                momentum=self.MOMENTUM, reward_scaling=self.REWARD_SCALING, reward_scaling_update=self.REWARD_SCALING_UPDATE
            )
        self.replay_memory = QAgentNN.ReplayMemory(memory_size, batch_size, dim_state, len(self.ACTIONS), num_buffer)
        self.freeze_counter = self.FREEZE_PERIOD - 1
        self.update_counter = self.UPDATE_PERIOD - 1
        self.rs_counter = self.RS_PERIOD - 1

    def reset(self, foget_table=False, new_table=None, foget_memory=False):
        self.last_state = None
        self.last_action = None
        self.freeze_counter = self.FREEZE_PERIOD - 1
        self.update_counter = self.UPDATE_PERIOD - 1
        self.rs_counter = self.RS_PERIOD - 1
        if foget_table:
            if isinstance(new_table, lasagne.layers.Layer):
                self.q_table = new_table
            else:
                raise ValueError("Please pass in a NN as new table")
        if foget_memory:
            self.ReplayMemory.reset()

    def transition_(self, observation, last_reward):
        """Update replay memory with new experience
        Consider the "memory state" as part of the agent state
        """
        state = observation
        last_state = self.last_state
        last_action = self.last_action

        # update current experience into replay memory
        if last_state is not None and state is not None:
            idx_action = self.ACTIONS.index(last_action)
            self.replay_memory.update(last_state, idx_action, last_reward, state)
        return state

    def reinforce_(self, state, last_reward):
        """Train the network periodically with past experiences
        Periodically train the network with random samples from the replay memory. Freeze the network parameters when in
        non-training epochs.

        Will not update the network if the state or reward passes in is None or the replay memory is yet to be filled up.

        Parameters
        ----------
        state : current agent state
        last_reward : reward from last action

        Returns : training loss
        -------

        """
        # update network if not frozen or dry run: sample memory and train network
        if state is None:
            if self.verbose > 0:
                print "  QAgentNN: ",
                print "state is None, agent not updated."
            return None
        elif last_reward is None:
            if self.verbose > 0:
                print "  QAgentNN: ",
                print "last_reward is None, agent not updated."
            return None
        elif not self.replay_memory.isfilled():
                if self.verbose > 0:
                    print "  QAgentNN: ",
                    print "unfull memory."
        else:
            loss = None

            if self.verbose > 0:
                print "  QAgentNN: ",
                # print "update counter {}, freeze counter {}.".format(self.update_counter, self.freeze_counter)
                print "update counter {}, freeze counter {}, rs counter {}.".format(
                    self.update_counter, self.freeze_counter, self.rs_counter)

            if self.update_counter == 0:
                last_states, last_actions, last_rewards, states = self.replay_memory.sample_batch()
                loss = self.update_table_(last_states, last_actions, last_rewards, states)
                self.update_counter = self.UPDATE_PERIOD - 1

                if self.verbose > 0:
                    print "  QAgentNN: ",
                    print "update loss is {}, reward_scaling is {}".format(loss, self.REWARD_SCALING)
                if self.verbose > 1:
                    freq = itemfreq(last_actions)
                    print "    QAgentNN: ",
                    print "batch action distribution: {}".format(
                        {self.ACTIONS[int(freq[i, 0])]: 1.0 * freq[i, 1] / self.BATCH_SIZE for i in range(freq.shape[0])}
                    )
            else:
                self.update_counter -= 1

            return loss

    def act_(self, state):
        # Escalate to QAgent.act_(). Pass None state if memory is not full to invoke random action.
        return super(QAgentNN, self).act_(state if self.is_memory_filled() else None)

    def update_table_(self, last_state, last_action, reward, current_state):
        loss = self.fun_train_qnn(
            self.rescale_state(last_state), last_action, reward, self.rescale_state(current_state)
        )

        if self.freeze_counter == 0:
            self.fun_clone_target()
            self.freeze_counter = self.FREEZE_PERIOD - 1
        else:
            self.freeze_counter -= 1
        if self.REWARD_SCALING_UPDATE=='adaptive':
            if self.rs_counter == 0:
                loss = self.fun_adapt_rs(
                    self.rescale_state(last_state), last_action, reward, self.rescale_state(current_state)
                )
                self.rs_counter = self.RS_PERIOD - 1
            else:
                self.rs_counter -= 1

        self.REWARD_SCALING = self.fun_rs_lookup()

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

    def init_fun_(self, dim_state, batch_size, gamma, learning_rate, momentum, reward_scaling, reward_scaling_update):
        """Define and compile function to train and evaluate network
        :param net: Lasagne output layer
        :param dim_state: dimensions of a single state tensor
        :param batch_size:
        :param gamma: future reward discount factor
        :param learning_rate:
        :param momentum:
        :param reward_scaling:
        :param reward_scaling_update:
        :return:
        """
        if len(dim_state) != 3:
            raise ValueError("We only support 3 dimensional states.")

        # inputs
        old_states, new_states = T.tensor4s('old_states', 'new_states')   # (BATCH_SIZE, MEMORY_LENGTH, DIM_STATE[0], DIM_STATE[1])
        actions = T.ivector('actions')           # (BATCH_SIZE, 1)
        rewards = T.vector('rewards')            # (BATCH_SIZE, 1)
        rs = shared(value=reward_scaling*1.0, name='reward_scaling')

        # intermediates
        predict_q = lasagne.layers.get_output(layer_or_layers=self.qnn, inputs=old_states)
        predict_next_q = lasagne.layers.get_output(layer_or_layers=self.qnn_target, inputs=new_states)
        target_q = rewards/rs + gamma*T.max(predict_next_q, axis=1)

        # penalty
        singularity = 1+1e-3
        penalty = T.mean(
            1/T.pow(predict_q[T.arange(batch_size), actions]-singularity, 2) +
            1/T.pow(predict_q[T.arange(batch_size), actions]+singularity, 2) - 2)


        # outputs
        loss = T.mean((predict_q[T.arange(batch_size), actions] - target_q)**2) + (1e-5)*penalty

        # weight update formulas (mini-batch SGD with momentum)
        params = lasagne.layers.get_all_params(self.qnn, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=momentum)
        updates_rs = lasagne.updates.nesterov_momentum(loss, [rs], learning_rate=learning_rate, momentum=momentum)

        # functions
        fun_train_qnn = theano.function([old_states, actions, rewards, new_states], loss, updates=updates, allow_input_downcast=True)
        fun_adapt_rs = theano.function([old_states, actions, rewards, new_states], loss, updates=updates_rs, allow_input_downcast=True)

        def fun_clone_target():
            lasagne.layers.helper.set_all_param_values(
                self.qnn_target,
                lasagne.layers.helper.get_all_param_values(self.qnn)
            )

        fun_q_lookup = theano.function([old_states], predict_q, allow_input_downcast=True)
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


if __name__ == '__main__':
    maze = SimpleMaze()
    agent = QAgentNN(dim_state=(1, 1, 2), range_state=((((0, 3),(0, 4)),),), actions=maze.ACTIONS,
                     learning_rate=0.01,
                     reward_scaling=100.0, reward_scaling_update='adaptive', rs_period=2,
                     batch_size=100, update_period=10,
                     freeze_period=2, memory_size=1000,
                     alpha=0.5, gamma=0.5, explore_strategy='epsilon', epsilon=0.02, verbose=2)
    print "Maze and agent initialized!"

    # logging
    path = deque()  # path in this episode
    episode_reward_rates = []
    num_episodes = 0
    cum_reward = 0
    cum_steps = 0

    # repeatedly run episodes
    while True:
        maze.reset()
        agent.reset()
        action, _ = agent.observe_and_act(observation=None, last_reward=None)  # get and random action
        path.clear()
        episode_reward = 0
        episode_steps = 0
        episode_loss = 0

        # print '(',
        # interact and reinforce repeatedly
        while not maze.isfinished():
            new_observation, reward = maze.interact(action)
            action, loss = agent.observe_and_act(observation=new_observation, last_reward=reward)
            # print new_observation,
            # print action,
            # print agent.fun_rs_lookup(),
            path.append(new_observation)
            episode_reward += reward
            episode_steps += 1
            episode_loss += loss if loss else 0
        # print '):',
        print len(path),
        # print "{:.3f}".format(episode_loss),
        # print ""
        cum_steps += episode_steps
        cum_reward += episode_reward
        num_episodes += 1
        episode_reward_rates.append(episode_reward / episode_steps)
        if num_episodes % 100 == 0:
            print ""
            print num_episodes, cum_reward, cum_steps, 1.0 * cum_reward / cum_steps #, path
            cum_reward = 0
            cum_steps = 0
    win = 50
    # s = pd.rolling_mean(pd.Series([0]*win+episode_reward_rates), window=win, min_periods=1)
    # s.plot()
    # plt.show()

