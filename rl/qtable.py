from collections import deque, Hashable

from numpy import max, exp
from numpy.random import rand, randint, multinomial


class QAgent(object):
    """Table-based Q Learning Agent
    A table-based Q learning agent. Q function is represented with a look-up table. Updates value function with
    Bellman iteration. Support random exploration using epsilon- or soft-probability strategy.
    """
    def __init__(self, actions=None,
                 alpha=1.0, gamma=0.5,
                 epsilon=0.02, explore_strategy='epsilon',
                 verbose=0, **kwargs):
        """

        Parameters
        ----------
        actions : Legitimate actions. (List or Tuple)
        alpha : Learning rate. The weight for update q value. (float, between (0, 1])
        gamma : Discount factor. (float, between [0, 1))
        epsilon : exploration rate for epsilon-greedy exploration. (float, [0, 1))
        explore_strategy : 'epsilon' or 'soft_probability'
        verbose : verbosity level.
        kwargs :

        Returns
        -------

        """
        # static attributes
        self.ACTIONS = actions    # legitimate actions
        self.ALPHA = alpha        # learning rate
        self.GAMMA = gamma        # discount factor
        self.EPSILON = epsilon    # exploration probability for 'epsilon-greedy' strategy
        self.DEFAULT_QVAL = 0     # default initial value for Q table entries
        self.EXPLORE = explore_strategy
        self.verbose = verbose

        # private dynamic attributes
        self.__last_observation = None
        self.__last_state = None
        self.__last_action = None
        self.__q_table = {}

        return

    def observe_and_act(self, observation, last_reward=None):
        """A single reinforcement learning step
        Operational procedures:
            1. improve environment model(s) and translate observation to agent state.
            2. reinforce value/policy.
            3. choose actions.

        Parameters
        ----------
        observation : environment observation at current step.
        last_reward : reward for latest action.

        Returns
        -------
        """
        exp_obs = (self.__last_observation, self.__last_action, last_reward, observation)
        # Improve model and translate observation to agent state
        state = self.improve_translate_(*exp_obs)
        exp_state = (self.__last_state, self.__last_action, last_reward, state)
        # Improve value/policy given current state and last_reward
        update_result = self.reinforce_(*exp_state)
        # Choose action based on current state
        action = self.act_(state)

        # Update buffer
        self.__last_observation = observation
        self.__last_state = state
        self.__last_action = action

        return action, update_result

    def reset(self, foget_table=False):
        self.__last_observation = None
        self.__last_state = None
        self.__last_action = None
        if foget_table:
            self.__q_table = {}

        return

    def improve_translate_(self, last_observation, last_action, last_reward, observation):
        return observation

    def reinforce_(self, last_state, last_action, last_reward, state):
        """ Improve value function
        """
        if last_state is None or state is None or last_reward is None:
            update_result = None
        else:
            update_result = self.update_table_(last_state, last_action, last_reward, state)

        return update_result

    def update_table_(self, last_state, last_action, last_reward, state):
        """Update Q function
        Use off-policy Bellman iteration:
            Q_new(s, a) = (1-alpha)*Q_old(s, a) + alpha*(R + gamma*max_a'{Q_old(s', a')})
        """
        best_qval = max(self.lookup_table_(state))
        delta_q = last_reward + self.GAMMA * best_qval

        if not isinstance(last_state, Hashable):
            last_state = tuple(last_state.ravel())  # TODO: assume should be numpy array

        self.__q_table[(last_state, last_action)] = \
            (1-self.ALPHA)*(
                self.__q_table[(last_state, last_action)]
                if (last_state, last_action) in self.__q_table
                else self.DEFAULT_QVAL
            ) + self.ALPHA*delta_q

        return None

    def act_(self, state):
        """Choose an action based on current state.

        Support epsilon-greedy and soft_probability exploration strategies.
        """
        # if state cannot be internalized as state, random act
        if state is None:
            idx_action = randint(0, len(self.ACTIONS))
            if self.verbose > 0:
                print " "*4 + "QAgent.act_():",
                print "randomly choose action (None state)."

        # random explore with "epsilon" probability.
        elif self.EXPLORE == 'epsilon':
            if rand() < self.EPSILON:
                idx_action = randint(0, len(self.ACTIONS))
                if self.verbose > 0:
                    print " "*4 + "QAgent.act_():",
                    print "randomly choose action (Epsilon)."
            else:  # select the best action with "1-epsilon" prob., break tie randomly
                q_vals = self.lookup_table_(state)
                max_qval = max(q_vals)
                idx_best_actions = [i for i in range(len(q_vals)) if q_vals[i] == max_qval]
                idx_action = idx_best_actions[randint(0, len(idx_best_actions))]
                if self.verbose > 0:
                    print " "*4 + "QAgent.act_():",
                    print "choose best q among {} (Epsilon).".format(
                        {self.ACTIONS[i]: q_vals[i] for i in range(len(self.ACTIONS))}
                    )

        # soft probability
        elif self.EXPLORE == 'soft_probability':
                q_vals = self.lookup_table_(state)  # state = internal_state
                exp_q_vals = exp(q_vals)
                idx_action = multinomial(1, exp_q_vals/sum(exp_q_vals)).nonzero()[0][0]
                if self.verbose > 0:
                    print " "*4 + "QAgent.act_():",
                    print "choose best q among {} (SoftProb).".format(dict(zip(self.ACTIONS, q_vals)))
        else:
            raise ValueError('Unknown keyword for exploration strategy!')
        return self.ACTIONS[idx_action]

    def lookup_table_(self, state):
        """ return the q values of all ACTIONS at a given state
        """
        if not isinstance(state, Hashable):
            state = tuple(state.ravel())

        return [
            self.__q_table[(state, a)]
            if (state, a) in self.__q_table
            else self.DEFAULT_QVAL
            for a in self.ACTIONS
       ]


