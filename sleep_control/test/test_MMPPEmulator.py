import sys
sys.path.append('../../')
import json

import numpy as np
import pandas as pd

from sleep_control.traffic_emulator import MMPPEmulator
from rl.mmpp import MMPP

model = MMPP(n_components=2, n_iter=1, init_params='', verbose=False)
model.startprob_ = np.array([.5, .5])
model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
model.emissionrates_ = np.array([10.0, 0.0])

session_df = pd.DataFrame()
head = pd.datetime(year=2014, month=9, day=4, hour=1)
tail = pd.datetime(year=2014, month=9, day=4, hour=2)
time_step = pd.Timedelta(seconds=10)
te = MMPPEmulator(session_df=session_df, head_datetime=head, tail_datetime=tail, time_step=time_step, verbose=0, traffic_model=model)

# while True:
for i in range(0, 1000):
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

