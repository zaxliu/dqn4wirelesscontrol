from numpy.random import randint


class SimpleMaze:
    """A simple maze game with a single goal state
    This is a simple maze game for testing q-learning agents. The game is
    played on a two-dimensional ground, meaning the agent can move up, down,
    left, or right to move around. One of the grids is the goal state. Each
    time the agent hits the goal state, a game round is terminated. The grids
    by the boundary are considered "walls". The agent can choose to move
    towards the wall, however will only be hit back to remain at the same state.
    """
    def __init__(self, dims=None, goal_state=None, goal_reward=100, wall_reward=0, null_reward=0):
        self.ACTIONS = ['left', 'right', 'up', 'down']  # legitimate ACTIONS
        self.DIMS = (4, 5) if dims is None else dims                     # shape of the maze ground
        self.GOAL_STATE = (2, 2)  if goal_state is None else goal_state  # coordinate of the goal state
        self.GOAL_REWARD = goal_reward    # reward given at goal state
        self.WALL_REWARD = wall_reward    # reward given when hit the wall
        self.NULL_REWARD = null_reward    # reward given otherwise
        self.state = None         # the coordinate of the agent
        self.reset()

    def observe(self):
        """ Observe the current maze state
        """
        return self.state

    def interact(self, action):
        """Emit next state and reward based on current state and action

        Parameters
        ----------
        action : must be contailed in self.ACTIONS
        -------

        """
        next_state, reward = self.transition_(self.state, action)
        self.state = next_state
        return next_state, reward

    def transition_(self, current_state, action):
        """State transition and rewarding logic

        """
        if action == 'up':
            next_state, reward = (current_state, self.WALL_REWARD) if current_state[0] == 0 \
                else ((current_state[0]-1, current_state[1]), self.NULL_REWARD)
        elif action == 'down':
            next_state, reward = (current_state, self.WALL_REWARD) if current_state[0] == (self.DIMS[0] - 1) \
                else ((current_state[0]+1, current_state[1]), self.NULL_REWARD)
        elif action == 'left':
            next_state, reward = (current_state, self.WALL_REWARD) if current_state[1] == 0 \
                else ((current_state[0], current_state[1]-1), self.NULL_REWARD)
        elif action == 'right':
            next_state, reward = (current_state, self.WALL_REWARD) if current_state[1] == (self.DIMS[1] - 1) \
                else ((current_state[0], current_state[1]+1), self.NULL_REWARD)
        else:
            print 'I don\'t understand this action ({}), I\'ll stay.'.format(action)
            next_state, reward = current_state, self.NULL_REWARD
        reward = self.GOAL_REWARD if next_state == self.GOAL_STATE else reward
        return next_state, reward

    def reset(self):
        """Randomly throw the agent to a non-goal state

        """
        next_state = self.GOAL_STATE
        while next_state == self.GOAL_STATE:
            next_state = (randint(0, self.DIMS[0]), randint(0, self.DIMS[1]))
        self.state = next_state

    def isfinished(self):
        return self.state == self.GOAL_STATE
