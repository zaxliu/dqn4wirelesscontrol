import sys
sys.path.append('../../')  # add project home into search path

import time
import pandas as pd

from sleep_control.integration import Emulation
from sleep_control.traffic_emulator import TrafficEmulator
from sleep_control.traffic_server import TrafficServer
from sleep_control.controller import QController, DummyController
from rl.qtable import QAgent
from rl.qnn_theano import QAgentNN

pd.set_option('mode.chained_assignment', None)

# Setting up data
print "Reading data..."
session_df = pd.read_csv(filepath_or_buffer='../data/net_traffic_nonull_sample.dat', sep=',', names=['uid','location','startTime_unix','duration_ms','domainProviders','domainTypes','domains','bytesByDomain','requestsByDomain'])
session_df.index.name = 'sessionID'
session_df['endTime_unix'] = session_df['startTime_unix'] + session_df['duration_ms']
session_df['startTime_datetime'] = pd.to_datetime(session_df['startTime_unix'], unit='ms')  # convert start time to readible date_time strings
session_df['endTime_datetime'] = pd.to_datetime(session_df['endTime_unix'], unit='ms')
session_df['totalBytes'] = session_df['bytesByDomain'].apply(lambda x: x.split(';')).map(lambda x: sum(map(float, x)))  # sum bytes across domains
session_df['totalRequests'] = session_df['requestsByDomain'].apply(lambda x: x.split(';')).map(lambda x: sum(map(float, x)))  # sum requests across domains
session_df.sort(['startTime_datetime'], ascending=True, inplace=True)  # get it sorted
session_df['interArrivalDuration_datetime'] = session_df.groupby('location')['startTime_datetime'].diff()  # group-wise diff
session_df['interArrivalDuration_ms'] = session_df.groupby('location')['startTime_unix'].diff()  # group-wise diff

print "Setting up Emulation environment..."
# Setting up Emulation
te = TrafficEmulator(session_df=session_df, time_step=pd.Timedelta(minutes=30), verbose=1)
ts = TrafficServer(verbose=2)
# ACTIONS = [(s, c) for s in [True, False] for c in ['serve_all', 'queue_all', 'random_serve_and_queue']]
actions = [(True, None), (False, 'serve_all')]
# agent = QAgent(ACTIONS=ACTIONS, alpha=0.5, gamma=0.5, explore_strategy='epsilon', epsilon=0.1)
agent = QAgentNN(dim_state=(1, 1, 3), range_state=((((0, 10), (0, 10), (0, 10),),),),
                 learning_rate=0.01, reward_scaling=10, batch_size=100, freeze_period=50, memory_size=200, num_buffer=2,
                 actions=actions, alpha=0.5, gamma=0.5, explore_strategy='epsilon', epsilon=0.1,
                 verbose=2
                 )
c = QController(agent=agent)
emu = Emulation(te=te, ts=ts, c=c)

print "Emulation starting"
print
# run...
while emu.te.epoch is not None:
    # log time
    print "Epoch {}, ".format(emu.epoch),
    left = emu.te.head_datetime + emu.te.epoch*emu.te.time_step
    right = left + emu.te.time_step
    print "{} - {}".format(left.strftime("%Y-%m-%d %H:%M:%S"), right.strftime("%Y-%m-%d %H:%M:%S"))
    emu.step()
    print

