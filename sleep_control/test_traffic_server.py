import numpy as np
import pandas as pd
import json
from traffic_server import TrafficServer
pd.set_option('mode.chained_assignment', None)

# Initialize
ts = TrafficServer()

# sleep all time
byteSent_req_domain_list = [{'111.com': {1: 2}, '222.com': {1: 3}},
                            {'333.com': {1: 2}, '444.com': {1: 3}},
                            {'555.com': {1: 2}, '666.com': {1: 3}}
                            ]
traffic_df = pd.DataFrame(columns=['sessionID', 'uid', 'bytesSent_per_request_per_domain'],
                          data={'sessionID': pd.Series([0, 1, 2]),
                                'uid': pd.Series([0, 1, 2]),
                                'bytesSent_per_request_per_domain': pd.Series([json.dumps(brd) for brd in byteSent_req_domain_list])
                                }
                          )

ts.feed_traffic(traffic_df=traffic_df)
print ts.observe()
control = False, 'queue_all'
print ts.get_service_and_cost(control=control)

byteSent_req_domain_list = [{'111.com': {2: 2}, '222.com': {2: 3}},
                            {'333.com': {2: 2}, '444.com': {2: 3}},
                            {'555.com': {2: 2}, '666.com': {2: 3}}
                            ]
traffic_df = pd.DataFrame(columns=['sessionID', 'uid', 'bytesSent_per_request_per_domain'],
                          data={'sessionID': pd.Series([0, 1, 2]),
                                'uid': pd.Series([0, 1, 2]),
                                'bytesSent_per_request_per_domain': pd.Series([json.dumps(brd) for brd in byteSent_req_domain_list])
                                }
                          )
ts.feed_traffic(traffic_df=traffic_df)
print ts.observe()
control = False, 'serve_all'
print ts.get_service_and_cost(control=control)
