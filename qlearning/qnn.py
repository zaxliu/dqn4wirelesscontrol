import numpy as np
from qtable import QAgent
import theano
import theano.tensor as T
import lasagne


class QAgentNN(QAgent):
    def __init__(self, dim_state, actions,  # basic properties
                 net=None, freeze_period=0, memory_size=100,  # nn related property
                 batch_size=100, learning_rate=0.01,
                 alpha=1.0, gamma=0.5, epsilon=0.0, init_state=None  # ql related property
                 ):
        super(QAgentNN, self).__init__(actions=actions, alpha=alpha,
                                       gamma=gamma, epsilon=epsilon,
                                       init_state=init_state)
        self.DIM_STATE = dim_state
        self.FREEZE_PERIOD = freeze_period
        self.MEMORY_SIZE = memory_size
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        # set q table as a NN
        if not net:
            net = QAgentNN.build_qnn_(None, tuple([None]+list(self.DIM_STATE)), len(self.ACTIONS))
        self.q_table = net
        self.fun_train_batch, self.fun_q_lookup = QAgentNN.init_fun_(self.q_table, self.DIM_STATE,
                                                                     self.BATCH_SIZE, self.GAMMA,
                                                                     self.LEARNING_RATE)
        self.replay_memory = self.init_replay_memory_()

    def update_table_(self, new_state, reward):
        # put current experience into memory
        # sample experience
        # formulate training samples
        # train network every freeze period
        pass

    def lookup_table_(self, state):
        state_var = np.zeros(tuple([1]+list(self.DIM_STATE)))
        state_var[0, :] = np.array(state)
        return self.fun_q_lookup(state_var).tolist()

    @staticmethod
    def build_qnn_(input_var=None, input_shape=None, num_outputs=None):
        if input_shape is None or num_outputs is None:
            raise ValueError('State or Action dimension not given!')
        l_in = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
        l_hid = lasagne.layers.DenseLayer(
            l_in, num_units=500,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        l_out = lasagne.layers.DenseLayer(
            l_hid, num_units=num_outputs,
            nonlinearity=lasagne.nonlinearities.sigmoid)
        return l_out

    @staticmethod
    def init_fun_(net, dim_state, batch_size, gamma, learning_rate):
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
        target_q = rewards + gamma*T.max(predict_next_q, axis=1)
        # outputs
        loss = T.mean((predict_q[T.arange(batch_size), actions] - target_q)**2)
        # weight update formulas (sgd)
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.sgd(loss, params, learning_rate=learning_rate)
        # functions
        fun_train_batch = theano.function([old_states, actions, rewards, new_states], loss, updates=updates)  # training function for one batch
        fun_q_lookup = theano.function([old_states], predict_q)
        return fun_train_batch, fun_q_lookup

    def init_replay_memory_(self):
        replay_memory = None
        return replay_memory

    def update_replay_memory_(self):
        pass

    def sample_replay_memory_(self):
        batch = None
        return batch

