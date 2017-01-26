import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
import json
from sleep_control.traffic_server import TrafficServer
pd.set_option('mode.chained_assignment', None)

# Initialize
ts = TrafficServer()

byteSent_req_domain_list = [{'111.com': {1: 2}, '222.com': {1: 3}},
                            {'333.com': {1: 2}, '444.com': {1: 3}},
                            {'555.com': {1: 2}, '666.com': {1: 3}}
                            ]
traffic_df = pd.DataFrame(columns=['sessionID', 'uid', 'bytesSent_per_request_domain'],
                          data={'sessionID': pd.Series([0, 1, 2]),
                                'uid': pd.Series([0, 1, 2]),
                                'bytesSent_per_request_domain': pd.Series([json.dumps(brd) for brd in byteSent_req_domain_list])
                                }
                          )

print ts.observe(traffic_df=traffic_df)
control = False, 'queue_all'
print ts.get_service_and_cost(control=control)

print ts.observe(traffic_df=traffic_df)
control = False, 'serve_all'
print ts.get_service_and_cost(control=control)

print ts.observe(traffic_df=traffic_df)
control = False, 'random_serve_and_queue'
print ts.get_service_and_cost(control=control)
