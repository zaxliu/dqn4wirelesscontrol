import sys
sys.path.append("../../")

import numpy as np
import pandas as pd
import json
from sleep_control.traffic_emulator import TrafficEmulator, PoissonEmulator

pd.set_option('mode.chained_assignment', None)

# Setting up data
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

# ============== Initialization ================
# Empty session_df
print "=======Initialization: Empty session_df======="
try:
    te = TrafficEmulator()  # should raise ValueError
except ValueError:
    pass
finally:
    pass

# Default values
print "=======Initialization: Default values======="
te = TrafficEmulator(session_df)
print te.time_step
print te.head_datetime
print te.tail_datetime
print te.verbose

# Verbose
print "=======Initialization: Verbose======="
te = TrafficEmulator(session_df, verbose=1)

# Head and tail datetime
print "=======Initialization: Head and tail datetime======="
head, tail = pd.datetime(year=2014, month=9, day=5), pd.datetime(year=2014, month=9, day=3)
try:
    te = TrafficEmulator(session_df, head_datetime=head, tail_datetime=tail, time_step=pd.Timedelta(days=0.5))
except ValueError:
    pass

# ============== Traffic & Service ================
# Head and tail range:
# Datetime range larger than dataset. should observe empty traffic at first, and warning in the end.
print "=======Traffic & Service: datetime range======="
head, tail, time_step = pd.datetime(year=2014, month=9, day=3), pd.datetime(year=2014, month=9, day=7), pd.Timedelta(days=0.5)
te = TrafficEmulator(session_df, head_datetime=head, tail_datetime=tail, time_step=time_step)
for i in range(0, 10):
    print "{} to {}".format(head+i*time_step, head+(i+1)*time_step)
    t = te.generate_traffic()
    if t is not None:
        print t.index
    else:
        pass
    print "Reward = {}".format(te.serve_and_reward(service_df=pd.DataFrame()))

# No service:
# Provide no service for all sessions. Should observe active session persist during its period.
# And reward = -1 * (# sent sessions)
print "=======Traffic & Service: no service======="
head = pd.datetime(year=2014, month=9, day=4, hour=1)
tail = pd.datetime(year=2014, month=9, day=4, hour=1, minute=1, second=50)
time_step = pd.Timedelta(seconds=10)
te = TrafficEmulator(session_df, head_datetime=head, tail_datetime=tail, time_step=time_step, verbose=1)
for i in range(0, 10):
    print "{} to {}".format(head+i*time_step, head+(i+1)*time_step)
    t = te.generate_traffic()
    if t is not None:
        print t.index
    else:
        pass
    service_df = pd.DataFrame(columns=['sessionID', 'service_per_request_domain'], index=t.index if t is not None else pd.Index([]))
    service_df['service_per_request_domain'] = json.dumps({})
    service_df['sessionID'] = t['sessionID']
    print te.serve_and_reward(service_df=service_df)

# Full service
print "=======Traffic & Service: full service======="
head = pd.datetime(year=2014, month=9, day=4, hour=1)
tail = pd.datetime(year=2014, month=9, day=4, hour=1, minute=1, second=50)
time_step = pd.Timedelta(seconds=10)
te = TrafficEmulator(session_df, head_datetime=head, tail_datetime=tail, time_step=time_step,verbose=0)
for i in range(0, 10):
    print "{} to {}".format(head+i*time_step, head+(i+1)*time_step)
    t = te.generate_traffic()
    if t is not None:
        print t
        service_df = pd.DataFrame(columns=['sessionID', 'service_per_request_domain'], index=t.index)

        for idx in t.index:
            bytesSent_req_domain = json.loads(t.loc[idx, 'bytesSent_per_request_domain'])
            service_req_domain = {}
            for domain in bytesSent_req_domain:
                for reqID in bytesSent_req_domain[domain]:
                    if domain not in service_req_domain:
                        service_req_domain[domain] = {}
                    service_req_domain[domain][int(reqID)] = 'serve'
            service_df.loc[idx, 'service_per_request_domain'] = json.dumps(service_req_domain)
            service_df.loc[idx, 'sessionID'] = t.loc[idx, 'sessionID']
    else:
        service_df = pd.DataFrame(columns=['sessionID', 'service_per_request_domain'], index=pd.Index([]))
    print te.serve_and_reward(service_df=service_df)


# Partial service
print "=======Traffic & Service: random service======="
head = pd.datetime(year=2014, month=9, day=4, hour=1)
tail = pd.datetime(year=2014, month=9, day=4, hour=1, minute=1, second=50)
time_step = pd.Timedelta(seconds=10)
te = TrafficEmulator(session_df, head_datetime=head, tail_datetime=tail, time_step=time_step,verbose=0)
for i in range(0, 10):
    print "{} to {}".format(head+i*time_step, head+(i+1)*time_step)
    t = te.generate_traffic()
    if t is not None:
        print t
        service_df = pd.DataFrame(columns=['sessionID', 'service_per_request_domain'], index=t.index)

        for idx in t.index:
            bytesSent_req_domain = json.loads(t.loc[idx, 'bytesSent_per_request_domain'])
            service_req_domain = {}
            for domain in bytesSent_req_domain:
                for reqID in bytesSent_req_domain[domain]:
                    if domain not in service_req_domain:
                        service_req_domain[domain] = {}
                    r = np.random.rand()
                    if r < 1.0/3:
                        service_req_domain[domain][int(reqID)] = 'serve'
                    elif r < 2.0/3:
                        service_req_domain[domain][int(reqID)] = 'queue'
                    else:
                        service_req_domain[domain][int(reqID)] = 'reject'
            service_df.loc[idx, 'service_per_request_domain'] = json.dumps(service_req_domain)
            service_df.loc[idx, 'sessionID'] = t.loc[idx, 'sessionID']
    else:
        service_df = pd.DataFrame(columns=['sessionID', 'service_per_request_domain'], index=pd.Index([]))
    print service_df
    print te.serve_and_reward(service_df=service_df)

# Poission Emulator
print "=======Poisson======="
head = pd.datetime(year=2014, month=9, day=4, hour=1)
tail = pd.datetime(year=2014, month=9, day=4, hour=1, minute=1, second=50)
time_step = pd.Timedelta(seconds=10)
te = PoissonEmulator(session_df=session_df, head_datetime=head, tail_datetime=tail, time_step=time_step, verbose=0, mu=3)
for i in range(0, 10):
    print "{} to {}".format(head+i*time_step, head+(i+1)*time_step)
    t = te.generate_traffic()
    if t is not None:
        print t
        service_df = pd.DataFrame(columns=['sessionID', 'service_per_request_domain'], index=t.index)

        for idx in t.index:
            bytesSent_req_domain = json.loads(t.loc[idx, 'bytesSent_per_request_domain'])
            service_req_domain = {}
            for domain in bytesSent_req_domain:
                for reqID in bytesSent_req_domain[domain]:
                    if domain not in service_req_domain:
                        service_req_domain[domain] = {}
                    r = np.random.rand()
                    if r < 1.0/3:
                        service_req_domain[domain][int(reqID)] = 'serve'
                    elif r < 2.0/3:
                        service_req_domain[domain][int(reqID)] = 'queue'
                    else:
                        service_req_domain[domain][int(reqID)] = 'reject'
            service_df.loc[idx, 'service_per_request_domain'] = json.dumps(service_req_domain)
            service_df.loc[idx, 'sessionID'] = t.loc[idx, 'sessionID']
    else:
        service_df = pd.DataFrame(columns=['sessionID', 'service_per_request_domain'], index=pd.Index([]))
    print service_df
    print te.serve_and_reward(service_df=service_df)
