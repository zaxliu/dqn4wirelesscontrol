import os
from collections import deque
import numpy as np
import theano
import theano.tensor as T
import lasagne
from qtable import QAgent, SimpleMaze


class QAgentNN(QAgent):
    def __init__(self, dim_state, range_state,  # basics
                 net=None, batch_size=100, learning_rate=0.01, momentum=0.9,  # nn related
                 reward_scaling=1, freeze_period=0,
                 memory_size=500,  # replay memory related
                 **kwargs):
        super(QAgentNN, self).__init__(**kwargs)

        self.DIM_STATE = dim_state  # mush be in form (d1, d2, d3), i.e. three dimensions
        self.STATE_MEAN = np.zeros(self.DIM_STATE)
        self.STATE_MAG = np.ones(self.DIM_STATE)
        if range_state:
            self.STATE_MEAN = (np.array(range_state)[:, :, :, 1]+np.array(range_state)[:, :, :, 0])/2.0  # upper bound on DIMS
            self.STATE_MAG = (np.array(range_state)[:, :, :, 1]-np.array(range_state)[:, :, :, 0])/2.0  # lower bound on DIMS
        self.FREEZE_PERIOD = freeze_period
        self.MEMORY_SIZE = memory_size
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.MOMENTUM = momentum
        self.REWARD_SCALING = reward_scaling
        # set q table as a NN
        if not net:
            net = QAgentNN.build_qnn_(None, tuple([None]+list(self.DIM_STATE)), len(self.ACTIONS))
        self.q_table = net
        self.fun_train_batch, self.fun_q_lookup = QAgentNN.init_fun_(self.q_table, self.DIM_STATE,
                                                                     self.BATCH_SIZE, self.GAMMA,
                                                                     self.LEARNING_RATE, self.MOMENTUM,
                                                                     self.REWARD_SCALING)
        self.replay_memory = QAgentNN.ReplayMemory(memory_size, batch_size, dim_state, len(self.ACTIONS))
        self.freeze_counter = 0

    def reset(self, foget_table=False, new_table=None, foget_memory=False):
        self.last_state = None
        self.last_action = None
        self.freeze_counter = 0
        if foget_table:
            if isinstance(new_table, lasagne.layers.Layer):
                self.q_table = new_table
            else:
                raise ValueError("Please pass in a NN as new table")
        if foget_memory:
            self.ReplayMemory.reset()

    def transition_(self, observation, reward):
        state = observation
        last_state = self.last_state
        last_action = self.last_action
        # update current experience into replay memory
        if last_state is not None and state is not None:
            idx_action = self.ACTIONS.index(last_action)
            self.replay_memory.update(last_state, idx_action, reward, state)
        return state

    def reinforce_(self, state, reward):
        # update network if not frozen or dry run: sample memory and train network
        if state is None:
            if self.verbose > 0:
                print "  QAgentNN: ",
                print "state is None, agent not updated."
            return None
        else:
            loss = None
            if (self.freeze_counter % self.FREEZE_PERIOD) == 0 and self.replay_memory.isfilled():
                last_state, last_action, reward, state = self.replay_memory.sample_batch()
                loss = self.update_table_(last_state, last_action, reward, state)
                self.freeze_counter = 0
                if self.verbose > 0:
                    print "  QAgentNN: ",
                    print "update loss is {} at {}".format(loss, self.freeze_counter)
            elif not self.replay_memory.isfilled():
                if self.verbose > 0:
                    print "  QAgentNN: ",
                    print "unfull memory."
            else:
                if self.verbose > 0:
                    print "  QAgentNN: ",
                    print "frozen net at counter {}.".format(self.freeze_counter)
            self.freeze_counter += 1

            return loss

    def update_table_(self, last_state, last_action, reward, current_state):
        loss = self.fun_train_batch(self.rescale_state(last_state), last_action, reward, self.rescale_state(current_state))
        return loss

    def lookup_table_(self, state):
        state_var = np.zeros(tuple([1]+list(self.DIM_STATE)), dtype=np.float32)
        state_var[0, :] = state
        return self.fun_q_lookup(self.rescale_state(state_var)).ravel().tolist()

    def is_memory_filled(self):
        return  self.replay_memory.isfilled()

    def rescale_state(self, states):
        return (states-self.STATE_MEAN)/self.STATE_MAG

    @staticmethod
    def build_qnn_(input_var=None, input_shape=None, num_outputs=None):
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

    @staticmethod
    def init_fun_(net, dim_state, batch_size, gamma, learning_rate, momentum, reward_scaling=1):
        """Define and compile function to train and evaluate network
        :param net:
        :param dim_state:
        :param batch_size:
        :param gamma:
        :param learning_rate:
        :param momentum:
        :return:
        """
        if len(dim_state) != 3:
            raise ValueError("Currently only support 3 dimensional states")
        # inputs
        old_states, new_states = T.tensor4s('old_states', 'new_states')   # (BATCH_SIZE, MEMORY_LENGTH, DIM_STATE[0], DIM_STATE[1])
        actions = T.ivector('actions')           # (BATCH_SIZE, 1)
        rewards = T.vector('rewards')            # (BATCH_SIZE, 1)
        # intemediate
        network = net
        predict_q = lasagne.layers.get_output(layer_or_layers=network, inputs=old_states)
        predict_next_q = lasagne.layers.get_output(layer_or_layers=network, inputs=new_states)
        target_q = rewards/reward_scaling + gamma*T.max(predict_next_q, axis=1)
        # outputs
        loss = T.mean((predict_q[T.arange(batch_size), actions] - target_q)**2)
        # weight update formulas (sgd)
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=momentum)
        # functions
        fun_train_batch = theano.function([old_states, actions, rewards, new_states], loss, updates=updates, allow_input_downcast=True)  # training function for one batch
        fun_q_lookup = theano.function([old_states], predict_q, allow_input_downcast=True)
        return fun_train_batch, fun_q_lookup

    class ReplayMemory(object):
        """Replay memory class
        """
        def __init__(self, memory_size, batch_size, dim_state, num_actions):
            self.MEMORY_SIZE = memory_size
            self.BATCH_SIZE = batch_size
            self.DIM_STATE = dim_state
            self.NUM_ACTIONS = num_actions

            self.buffer_old_state = np.zeros(tuple([memory_size]+list(self.DIM_STATE)), dtype=np.float32)
            self.buffer_action = np.zeros((memory_size, ), dtype=np.int32)
            self.buffer_reward = np.zeros((memory_size, ), dtype=np.float32)
            self.buffer_new_state = np.zeros(tuple([memory_size]+list(self.DIM_STATE)), dtype=np.float32)

            self.top = 0
            self.filled = False

        def update(self, old_state, idx_action, reward, new_state):
            top = self.top
            self.buffer_old_state[top, :] = old_state
            self.buffer_action[top] = idx_action
            self.buffer_reward[top] = reward
            self.buffer_new_state[top, :] = new_state
            if not self.filled:
                self.filled |= (top == (self.MEMORY_SIZE-1))
            self.top = (top+1) % self.MEMORY_SIZE

        def sample_batch(self):
            batch_idx = np.random.randint(0, self.MEMORY_SIZE, (self.BATCH_SIZE,))
            return (self.buffer_old_state[batch_idx, :],
                    self.buffer_action[batch_idx],
                    self.buffer_reward[batch_idx],
                    self.buffer_new_state[batch_idx, :])

        def isfilled(self):
            return self.filled

        def reset(self):
            self.top = 0
            self.filled = False


if __name__ == '__main__':
    maze = SimpleMaze()
    agent = QAgentNN(dim_state=(1, 1, 2), range_state=((((0, 3),(0, 4)),),), actions=maze.actions,
                     learning_rate=0.01, reward_scaling=100, batch_size=100,
                     freeze_period=20, memory_size=1000,
                     alpha=0.5, gamma=0.5, explore_strategy='epsilon', epsilon=0.02)
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
        action, _ = agent.observe_and_act(observation=None, reward=None)  # get and random action
        path.clear()
        episode_reward = 0
        episode_steps = 0
        episode_loss = 0

        # interact and reinforce repeatedly
        while not maze.isfinished():
            new_observation, reward = maze.interact(action)
            action, loss = agent.observe_and_act(observation=new_observation, reward=reward)
            # print action,
            # print new_observation,
            path.append(new_observation)
            episode_reward += reward
            episode_steps += 1
            episode_loss += loss if loss else 0
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

