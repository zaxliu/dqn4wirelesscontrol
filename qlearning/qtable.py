from collections import deque, Hashable

from numpy import max, exp
from numpy.random import rand, randint, multinomial

from qlearning.simple_envs import SimpleMaze


class QAgent(object):
    """Table-based Q Learning Agent
    A table-based Q learning agent. Updates table with canonical Bellman iteration method. Random exploration using
    epsilon- or soft-probability strategy.

    Train the agent with QAgent.observer_and_act(). Use the current observation and the reward from last action. Reset
    agent state with QAgent.reset(), set forget_table = True to drop the learned table.
    """
    def __init__(self, actions=None, alpha=1.0, gamma=0.5, epsilon=0.02, explore_strategy='epsilon', verbose=0, **kwargs):
        """

        Parameters
        ----------
        actions : Legitimate actions. (List or Tuple)
        alpha : Learning rate. The weight for update q value. (float, between (0, 1])
        gamma : Reward discount factor. (float, between [0, 1))
        epsilon : exploration rate for epsilon-greedy exploration. (float, [0, 1))
        explore_strategy : 'epsilon' or 'soft_probability'
        verbose : verbosity level.
        kwargs :

        Returns
        -------

        """
        super(QAgent, self).__init__(**kwargs)

        # static attributes
        if not actions:
            raise ValueError("Passed in None action list.")
        self.ACTIONS = actions    # legitimate actions
        self.ALPHA = alpha        # learning rate
        self.GAMMA = gamma        # discount factor
        self.EPSILON = epsilon    # exploration probability for 'epsilon-greedy' strategy
        self.DEFAULT_QVAL = 0     # default initial value for Q table entries
        self.EXPLORE = explore_strategy
        self.verbose = verbose

        # dynamic attributes
        self.last_state = None
        self.last_action = None
        self.q_table = {}

    def observe_and_act(self, observation, last_reward=None):
        """A single reinforcement learning step
        Pass in the current observation and reward from last action. First try to internalize the observation as an
        agent state, then reinforce the agent with current experience, finally perform action based on current experience.

        The internalization of observation is done by the transition_() method. This method is expected to be materialized
        in the child classes. If not, the state will simply be the current observation.
        """
        # Internalize observation and last_reward
        try:  # update agent state if a transition_() method is provided
            state = self.transition_(observation=observation, last_reward=last_reward)
        except AttributeError:  # otherwise use observation as agent state
            state = observation

        # Improve agent given current state and last_reward
        update_result = self.reinforce_(state=state, last_reward=last_reward)

        # Choose action based on current state
        action = self.act_(state=state)

        # Update
        self.last_state = state
        self.last_action = action
        return action, update_result

    def reset(self, foget_table=False):
        self.last_state = None
        self.last_action = None
        if foget_table:
            self.q_table = {}

    def reinforce_(self, state, last_reward):
        """ Improve agent based on current experience (last_state, last_action, last_reward, state)
        """
        last_state = self.last_state
        last_action = self.last_action
        if last_state is None or state is None or last_reward is None:
            update_result = None
        else:
            update_result = self.update_table_(last_state, last_action, last_reward, state)
        return update_result

    def update_table_(self, last_state, last_action, reward, current_state):
        """Update Q table using Bellman iteration
        """
        best_qval = max(self.lookup_table_(current_state))
        if not isinstance(last_state, Hashable):
            last_state = tuple(last_state.ravel())  # passed in numpy array
        delta_q = reward + self.GAMMA * best_qval
        self.q_table[(last_state, last_action)] = \
            self.ALPHA * delta_q + (1 - self.ALPHA) * (self.q_table[(last_state, last_action)]
                                                       if (last_state, last_action) in self.q_table else self.DEFAULT_QVAL)
        return None

    def act_(self, state):
        """Choose an action based on current state.
        """
        if state is None:
            idx_action = randint(0, len(self.ACTIONS))  # if state cannot be internalized as state, random act
            if self.verbose > 0:
                print "  QAgent: ",
                print "randomly choose action {} (None state).".format(self.ACTIONS[idx_action])
        elif self.EXPLORE == 'epsilon':
            if rand() < self.EPSILON:  # random exploration with "epsilon" prob.
                idx_action = randint(0, len(self.ACTIONS))
                if self.verbose > 0:
                    print "  QAgent: ",
                    print "randomly choose action (Epsilon)."
            else:  # select the best action with "1-epsilon" prob., break tie randomly
                q_vals = self.lookup_table_(state)
                max_qval = max(q_vals)
                idx_best_actions = [i for i in range(len(q_vals)) if q_vals[i] == max_qval]
                idx_action = idx_best_actions[randint(0, len(idx_best_actions))]
                if self.verbose > 0:
                    print "  QAgent: ",
                    print "choose best q among {} (Epsilon).".format(
                        {self.ACTIONS[i]: q_vals[i] for i in range(len(self.ACTIONS))}
                    )
        elif self.EXPLORE == 'soft_probability':
                q_vals = self.lookup_table_(state)  # state = internal_state
                exp_q_vals = exp(q_vals)
                idx_action = multinomial(1, exp_q_vals/sum(exp_q_vals)).nonzero()[0][0]
                if self.verbose > 0:
                    print "  QAgent: ",
                    print "choose best q among {} (SoftProb).".format(dict(zip(self.ACTIONS, q_vals)))
        else:
            raise ValueError('Unknown keyword for exploration strategy!')
        return self.ACTIONS[idx_action]

    def lookup_table_(self, state):
        """ return the q values of all ACTIONS at a given state
        """
        if not isinstance(state, Hashable):
            state = tuple(state.ravel())
        return [self.q_table[(state, a)] if (state, a) in self.q_table else self.DEFAULT_QVAL for a in self.ACTIONS]


if __name__ == "__main__":
    maze = SimpleMaze()
    agent = QAgent(actions=maze.ACTIONS, alpha=0.5, gamma=0.5, explore_strategy='epsilon', epsilon=0.1)
    # logging
    path = deque()  # path in this episode
    episode_reward_rates = []
    num_episodes = 0
    cum_reward = 0
    cum_steps = 0

    # repeatedly run episodes
    while True:
        # initialization
        maze.reset()
        agent.reset(foget_table=False)
        action, _ = agent.observe_and_act(observation=None, last_reward=None)  # get and random action
        path.clear()
        episode_reward = 0
        episode_steps = 0
        # interact and reinforce repeatedly
        while not maze.isfinished():
            new_observation, reward = maze.interact(action)
            action, _ = agent.observe_and_act(observation=new_observation, last_reward=reward)
            path.append(new_observation)
            episode_reward += reward
            episode_steps += 1

        cum_steps += episode_steps
        cum_reward += episode_reward
        num_episodes += 1
        episode_reward_rates.append(episode_reward / episode_steps)
        if num_episodes % 100 == 0:
            print num_episodes, len(agent.q_table), cum_reward, cum_steps, 1.0 * cum_reward / cum_steps#, path
            cum_reward = 0
            cum_steps = 0
    win = 50
    # s = pd.rolling_mean(pd.Series([0]*win+episode_reward_rates), window=win, min_periods=1)
    # s.plot()
    # plt.show()
