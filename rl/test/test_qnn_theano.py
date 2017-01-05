import sys
sys.path.append('../')

from collections import deque

from qnn_theano import QAgentNN
from simple_envs import SimpleMaze


maze = SimpleMaze()
agent = QAgentNN(dim_state=(1, 1, 2),           # SimpleMaze is a 2-D grid world 
            range_state=((((0, 3),(0, 4)),),),  # Default size is 4-by-5
            actions=maze.ACTIONS,
            learning_rate=0.01,
            reward_scaling=100.0, reward_scaling_update='adaptive', rs_period=2,  # 
            batch_size=100, update_period=10,
            freeze_period=2, memory_size=100,
            alpha=0.5, gamma=0.5, explore_strategy='epsilon', epsilon=0.02, verbose=0)
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
    agent.reset(foget_table=False, foget_memory=False)  # just reset counters
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
    print "{:.3f}".format(episode_loss),
    print ""
    cum_steps += episode_steps
    cum_reward += episode_reward
    num_episodes += 1
    episode_reward_rates.append(episode_reward / episode_steps)
    if num_episodes % 100 == 0:
        print ""
        print num_episodes, cum_reward, cum_steps, 1.0 * cum_reward / cum_steps #, path
        cum_reward = 0
        cum_steps = 0
        # agent.reset(foget_table=True, foget_memory=False)


