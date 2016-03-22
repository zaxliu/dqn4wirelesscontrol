from collections import deque
import numpy as np
from qtable import SimpleMaze
from qnn import QAgentNN


class PhiMixin(object):
    """Transform history of observations and actions into state
    This is a Mixin class, so some attributes or methods need to be implemented in other classes of the inheritance hierarchy
    """
    def __init__(self, phi_length, **kwargs):
        super(PhiMixin, self).__init__(**kwargs)
        self.PHI_LENGTH = phi_length
        self.phi_buffer = deque()

    def transition_(self, observation, reward):
        # Prepare values
        last_action = self.last_action
        if not last_action:
            return None
        idx_action = self.ACTIONS.index(last_action)
        # Update buffer
        state_slice = self.combine_(observation, idx_action)
        self.phi_buffer.append(state_slice)
        if len(self.phi_buffer) > self.PHI_LENGTH:
            self.phi_buffer.popleft()
        # Compose state
        state = np.array(self.phi_buffer)
        # Successive call
        if len(self.phi_buffer) < self.PHI_LENGTH:
            return None
        else:
            try:
                state = super(PhiMixin, self).transition_(observation=state, reward=reward)
            except AttributeError:
                pass
            finally:
                return state

    def combine_(self, observation, idx_action):
        # observation up to 2 dim, (a, b)
        # action single number
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


class QAgentNN_History(PhiMixin, QAgentNN):
    def __init__(self, **kwargs):
        super(QAgentNN_History, self).__init__(**kwargs)

if __name__ == '__main__':
    maze = SimpleMaze()
    slice_range = [(0, 3), (0, 4)] + zip([0]*len(maze.actions), [1]*len(maze.actions))
    phi_length = 5
    agent = QAgentNN_History(
        phi_length=phi_length,
        dim_state=(1, phi_length, 2+len(maze.actions)),
        range_state=[[slice_range]*5],
        actions=maze.actions,
        learning_rate=0.01, reward_scaling=100, batch_size=100,
        freeze_period=100, memory_size=1000,
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