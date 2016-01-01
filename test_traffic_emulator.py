import numpy as np
import pandas as pd
import json
from traffic_emulator import TrafficEmulator


pd.set_option('mode.chained_assignment', None)

# Extract data from file and save into pd.DataFrame
session_df = pd.read_csv(filepath_or_buffer='./data/net_traffic_nonull_sample.dat', sep=',',
                        names=['uid','location','startTime_unix','duration_ms','domainProviders','domainTypes','domains','bytesByDomain','requestsByDomain'])
session_df.index.name = 'sessionID'

# Create convenient columns
session_df['endTime_unix'] = session_df['startTime_unix'] + session_df['duration_ms']
session_df['startTime_datetime'] = pd.to_datetime(session_df['startTime_unix'], unit='ms')  # convert start time to readible date_time strings
session_df['endTime_datetime'] = pd.to_datetime(session_df['endTime_unix'], unit='ms')
session_df['totalBytes'] = session_df['bytesByDomain'].apply(lambda x: x.split(';')).map(lambda x: sum(map(float, x)))  # sum bytes across domains
session_df['totalRequests'] = session_df['requestsByDomain'].apply(lambda x: x.split(';')).map(lambda x: sum(map(float, x)))  # sum requests across domains
session_df.sort(['startTime_datetime'], ascending=True, inplace=True)  # get it sorted
session_df['interArrivalDuration_datetime'] = session_df.groupby('location')['startTime_datetime'].diff()  # group-wise diff
session_df['interArrivalDuration_ms'] = session_df.groupby('location')['startTime_unix'].diff()  # group-wise diff


te = TrafficEmulator(session_df=session_df, time_step=pd.Timedelta(days=1))
t = te.get_traffic()
i = 0
while(t is not None):
    print i
    print t
    te.serve(service=pd.DataFrame())
    t = te.get_traffic()
    i += 1
