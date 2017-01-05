import sys
sys.path.append('../')

from collections import deque

from qtable import QAgent
from simple_envs import SimpleMaze

maze = SimpleMaze()
agent = QAgent(actions=maze.ACTIONS, alpha=0.5, gamma=0.5, explore_strategy='epsilon', epsilon=0.01)

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
    # print len(path),

    cum_steps += episode_steps
    cum_reward += episode_reward
    num_episodes += 1
    episode_reward_rates.append(episode_reward / episode_steps)
    if num_episodes % 10 == 0:
        print num_episodes, len(agent._QAgent__q_table), cum_reward, cum_steps, 1.0 * cum_reward / cum_steps#, path
        cum_reward = 0
        cum_steps = 0
