import numpy as np
import pandas as pd
import json


class TrafficEmulator:
    """Emulate the traffic-generation behavior of a bunch of users

    Emulate the session dynamics in a network as the result of the interaction between a user population and
    a network server. Interaction is defined as discrete-time (traffic, service) pair. In each epoch the
    population emits some network traffic for the network traffic server, which conducts certain service.
    According to the service decision, traffic may be served, suspended, or denied. The users then emit a
    scalar reward for the service.

    Under the hood, the session dynamics is generated with a static network traffic dataset and some (simple)
    assumptions on the traffic generation and service response process. A session can either end, wait, or
    abort as a result of the service it receive. The emulator receives the service decision and updates
    its internal session states, based on which sessions are generated for later epochs.

    Parameters
    ----------
    session_df : pandas.DataFrame
        A DataFrame containing the raw session information. Columns...
    head_datetime : pandas.Datetime, default : minimal session start datetime in the dataset
        Start datetime for the emulation
    tail_datetime : pandas.Datetime, default : maximal session termination datetime in the dataset
        End datetime for the emulation
    time_step : pandas.Timedelta, default : Timedelta('1 Second')
        Time duration of each epoch.
    verbose : int, default : 0
        Verbosity level.
    """
    def __init__(self, session_df=None, head_datetime=None, tail_datetime=None, time_step=pd.Timedelta(seconds=1),
                 verbose=0):
        # Dataset =====================================================
        if session_df is None:
            print "TrafficEmulator Initialization: session_df passed in is empty or None."
            raise ValueError
        self.session_df = session_df  # Initialize session dataset
        # Time-related variables ======================================
        # starting datetime of the simulation, use datetime of first session arrival if not assigned
        self.head_datetime = head_datetime if head_datetime is not None \
            else self.session_df['startTime_datetime'].min()
        # end datetime of the simulation, use datetime of last session departure if not assigned
        self.tail_datetime = tail_datetime if tail_datetime is not None \
            else self.session_df['startTime_datetime'].max()
        if self.head_datetime > self.tail_datetime:  # sanity check
            print "head_datetime > tail_datetime"
            raise ValueError
        self.time_step = time_step
        # Verbosity ===================================================
        self.verbose = verbose  # verbosity level
        # Reset active session buffer and time counter ================
        self.active_sessions = None
        self.epoch = None
        self.reset()
        # Output ======================================================
        if verbose > 0:
            print "New TrafficEmulator with parameters:\n  " \
                  "head={}\n  tail={}\n  time_step={}\n  epoch={}\n  verbose={}".format(
                    self.head_datetime, self.tail_datetime, self.time_step, self.epoch, self.verbose)

    # Public Methods
    def generate_traffic(self):
        """Get traffic for this epoch, set epoch=None if run out of data
        First check the traffic dataset and add new sessions that are initiated during this epoch, add these new-coming
        sessions into the active session buffer. Then generate traffic (in term of how many bytes are sent in some
        request for some domain for each session).
        :return: traffic DataFrame
        """
        # ========================================================
        # 1. Add new incoming sessions to active session buffer
        # if already run out of data, suggest reset()
        if self.epoch is None:
            print "Run out of session data, please reset emulator."
            return None
        # if running out of data (left edge out of range), set epoch=None and return None
        left = self.head_datetime + self.epoch*self.time_step
        right = left + self.time_step
        if left >= self.tail_datetime:
            print "Reach tail_datetime, please reset dataset"
            self.epoch = None
            return None
        # else, get sessions initiated in this epoch, and append it to the active session buffer
        if self.verbose > 0:
            print "get_traffic(): locating incoming sessions."
        incoming_sessions = self.session_df.loc[(self.session_df['startTime_datetime'] >= left) &
                                                (self.session_df['startTime_datetime'] < right)]
        if self.verbose > 0:
            print "get_traffic(): appending incoming sessions to buffer."
        self.__append_to_active_sessions__(incoming_sessions)

        # ===========================================================
        # 2. Generate traffic according active session buffer content
        if self.verbose > 0:
            print "get_traffic(): generating traffic."
        traffic_df = self.__generate_requests__()
        if self.verbose > 0:
            print "get_traffic(): finished."
        return traffic_df

    def serve_and_reward(self, service_df):
        if self.epoch is not None:
            # Update active session buffer and emit reward according to the service provided
            service_reward = self.__evaluate_service__(service_df=service_df)
            # Increase timer by one epoch
            self.epoch += 1
        else:
            print "Ran out of data or reach tail, service ignored."
            service_reward = 0
        return service_reward

    def reset(self):
        # Active session buffer
        # addition columns :
        #   'endTime_datetime_updated' :
        #       When is the session expected to end? initialized with loged end time, may be modified based on service.
        #   'bytes_per_request_per_domain' :
        #       How many bytes goes into each requests of each domain? Stored as json-encoded dict, key is domain, value
        #       is the list of bytes for each request. This column is initiated here and will not be changed.
        #   'pendingReqID_per_domain' :
        #       Which request of each domain is ready to be sent but not yet sent. json encoded dict, key is domain,
        #       value is a list of request IDs. Intialized as [1, ...., total#reqeusts]
        #   'waitingReqID_per_domain' :
        #       Which request of each domain is sent but not yet served (queuing). json encoded dict, key is domain,
        #       value is a list of request IDs. Initialzed with empty dict.
        #   'servedReqID_per_domain' :
        #       Which request of each domain is successfully served. json encoded dict, key is domain,
        #       value is a list of request IDs. Initialzed with empty dict.
        #   'failedReqID_per_domain' :
        #       Which request of each domain is failed to be served. json encoded dict, key is domain,
        #       value is a list of request IDs. Initialzed with empty dict.
        self.active_sessions = pd.DataFrame(columns=self.session_df.columns | pd.Index([
            'endTime_datetime_updated', 'bytes_per_request_per_domain',
            'pendingReqID_per_domain', 'servedReqID_per_domain', 'waitingReqID_per_domain',
            'failedReqID_per_domain',
        ]))
        # Time counter
        self.epoch = 0

    # Private Methods
    def __append_to_active_sessions__(self, incoming_sessions):
        """Append incoming sessions to active session buffer and adjust format.

        :param incoming_sessions: incoming session DataFrame
        :return:
        """
        # Initialize internal state columns for each session ==============================================
        # 'endTime_datetime_updated' column
        incoming_sessions.loc[:, 'endTime_datetime_updated'] = incoming_sessions['endTime_datetime']
        for idx in incoming_sessions.index:
            # 'bytes_per_request_per_domain' column
            domains = incoming_sessions.loc[idx, 'domains'].split(';')
            bytes_domain = map(int, incoming_sessions.loc[idx, 'bytesByDomain'].split(';'))
            requests_domain = map(int, incoming_sessions.loc[idx, 'requestsByDomain'].split(';'))
            bytes_request_domain = dict(
                zip(domains,
                    [self.__distribute_bytes__(bytes_domain[i], requests_domain[i]) for i in range(len(domains))]
                    )
            )
            incoming_sessions.loc[idx, 'bytes_per_request_per_domain'] = json.dumps(bytes_request_domain)
            # 'pendingReqID_per_domain' column
            pendingReqID_domain = dict([(domain, range(len(bytes_request_domain[domain]))) for domain in bytes_request_domain])
            incoming_sessions.loc[idx, 'pendingReqID_per_domain'] = json.dumps(pendingReqID_domain)
            # 'waitingReqID_per_domain' column
            incoming_sessions.loc[idx, 'waitingReqID_per_domain'] = json.dumps({})
            # 'servedReqID_per_domain' column
            incoming_sessions.loc[idx, 'servedReqID_per_domain'] = json.dumps({})
            # 'failedReqID_per_domain' column
            incoming_sessions.loc[idx, 'failedReqID_per_domain'] = json.dumps({})
        # Append incoming session to active session buffer ==============================================
        self.active_sessions = self.active_sessions.append(incoming_sessions)
        return

    @staticmethod
    def __distribute_bytes__(bytes, requests):
        """ Distribute bytes among requests.

        Each byte independently goes into each request with equal probability. Assume B bytes and R
        requests, the bytes assigned to each request follow a multinomial distribution of B trails
        and R category.
        :param bytes: total bytes to assign
        :param requests: total requests to assign
        :return: a list of how many bytes per request
        """
        return list(np.random.multinomial(bytes, [1.0/requests]*requests))

    def __generate_requests__(self):
        """How many requests in each domain are issued in current epoch?

        Traffic drilling-down assumptions:
            Assume traffics from each session are generated independently. Each pending request is sent independently in
            the current epoch with probablity 1/epochs_left, i.e. uniformly distributed in the expected future.
        :return: a DataFrame with traffic info
        """
        traffic_df = pd.DataFrame(data=None, index=None, columns=['sessionID', 'uid', 'bytesSent_per_request_per_domain'])

        # Generate requests from each session in the active session buffer
        for sessionID in self.active_sessions.index:
            # extract info from active session buffer
            bytes_req_domain = json.loads(self.active_sessions.loc[sessionID, 'bytes_per_request_per_domain'])
            pendingReqID_domain = json.loads(self.active_sessions.loc[sessionID, 'pendingReqID_per_domain'])
            waitingReqID_domain = json.loads(self.active_sessions.loc[sessionID, 'waitingReqID_per_domain'])
            # servedReqID_domain = json.loads(self.active_sessions.loc[sessionID, 'servedReqID_per_domain'])
            # failedReqID_domain = json.loads(self.active_sessions.loc[sessionID, 'failedReqID_per_domain'])
            # how many epochs left for the current session?
            end_epoch = np.floor((self.active_sessions.loc[sessionID, 'endTime_datetime_updated'] - self.head_datetime) / self.time_step)
            epochs_left = end_epoch - self.epoch + 1
            # Which requests are sent in current epoch under each domain and how many bytes?
            bytesSent_req_domain = {}  # {domain1: [req1_id: req1_bytes, req2_id: req2_bytes, ...], domain2: ...}
            for domain in bytes_req_domain:
                # extract info lists from dicts
                pendingReqID = pendingReqID_domain[domain] if domain in pendingReqID_domain else []  # `else` case will not happen but put here for sanity.
                waitingReqID = waitingReqID_domain[domain] if domain in waitingReqID_domain else []
                # servedReqID = servedReqID_domain[domain] if domain in servedReqID_domain else []
                # failedReqID = failedReqID_domain[domain] if domain in failedReqID_domain else []
                # which requests to send?
                toSendFlag_pending = np.random.rand(len(pendingReqID)) < 1.0/epochs_left
                toSendReqID = np.array(pendingReqID)[toSendFlag_pending].tolist()
                # build traffic dict
                bytes_reqSend = np.array(bytes_req_domain[domain])[toSendReqID].tolist()
                if len(toSendReqID) > 0:
                    bytesSent_req_domain[domain] = dict(zip(toSendReqID, bytes_reqSend))
                # update info lists and write back to dicts
                pendingReqID = list(set(pendingReqID) - set(toSendReqID))  # take toSendReq out from pending list
                pendingReqID_domain[domain] = pendingReqID
                waitingReqID = list(set(waitingReqID).union(set(toSendReqID)))  # add toSendReq into waiting list
                waitingReqID_domain[domain] = waitingReqID
            # update active session buffer
            self.active_sessions.loc[sessionID, 'pendingReqID_per_domain'] = json.dumps(pendingReqID_domain)
            self.active_sessions.loc[sessionID, 'waitingReqID_per_domain'] = json.dumps(waitingReqID_domain)
            # generate current_traffic
            if len(bytesSent_req_domain) > 0:
                traffic_df = traffic_df.append(pd.DataFrame(
                    {'sessionID': sessionID,
                     'uid': self.active_sessions.loc[sessionID, 'uid'],
                     'bytesSent_per_request_per_domain': json.dumps(bytesSent_req_domain)}, index=[None]),
                    ignore_index=True)
        return traffic_df

    def __evaluate_service__(self, service_df):
        """Modify internal state columns in active session buffer according to the service provided

        Interaction assumptions:
            1: if a "waiting" request is served in the current epoch, then emit +1 and set state as "served".
            2: if a "waiting" request is queued in the current epoch, then emit -1 and keep the state as "waiting".
            3. if a "waiting" request is rejected in the current epoch, then emit -1 and set state as "pending" with
               probability 0.7 or emit -10 and set state as "failed" with probability 0.3. (0.7-persistent retransmission)
            4. if no instruction for a request, assume the server means to "queue" it, emit -1.
            5. if this is the last epoch, then assume all "waiting" requests "failed", emit -10.
        :param service_df: a data frame indicating the service for each traffic row
        :return: a scalar reward for the service
        """
        # Set initial reward to zero ===========================================================
        reward = 0
        # Interacte with each service row in service_df ========================================
        for sessionID in self.active_sessions.index:
            # extract info from service df
            idx = (service_df['sessionID'] == sessionID).nonzero()[0] if len(service_df) > 0 else np.array([])
            service_req_domain = json.loads(service_df.loc[idx[0], 'service_per_request_per_domain']) if len(idx) > 0 \
                else {}
            # extract info from active session buffer
            end_epoch = np.floor((self.active_sessions.loc[sessionID, 'endTime_datetime_updated'] - self.head_datetime) / self.time_step)
            pendingReqID_domain = json.loads(self.active_sessions.loc[sessionID, 'pendingReqID_per_domain'])
            waitingReqID_domain = json.loads(self.active_sessions.loc[sessionID, 'waitingReqID_per_domain'])
            servedReqID_domain = json.loads(self.active_sessions.loc[sessionID, 'servedReqID_per_domain'])
            failedReqID_domain = json.loads(self.active_sessions.loc[sessionID, 'failedReqID_per_domain'])
            # for each domain, check the services that 'waiting' requests received, and update active session
            # buffer accordingly
            for domain in waitingReqID_domain:
                # extract info from dicts
                pendingReqID = pendingReqID_domain[domain] if domain in pendingReqID_domain else []
                waitingReqID = waitingReqID_domain[domain] if domain in waitingReqID_domain else []
                servedReqID = servedReqID_domain[domain] if domain in servedReqID_domain else []
                failedReqID = failedReqID_domain[domain] if domain in failedReqID_domain else []
                service_req = service_req_domain[domain] if domain in service_req_domain else {}
                # indices of requests which are served, queued, and rejected...
                # Note: json converts key to string type
                serveReqID = [int(req_id) for req_id in service_req if service_req[req_id] == 'serve']
                queueReqID = [int(req_id) for req_id in service_req if service_req[req_id] == 'queue']
                rejectReqID = [int(req_id) for req_id in service_req if service_req[req_id] == 'reject']
                unattendReqID = [req_id for req_id in waitingReqID if str(req_id) not in service_req or
                                service_req[str(req_id)] not in ['serve', 'queue', 'reject']]
                # interactions & rewards...
                # case 1:
                waitingReqID = list(set(waitingReqID) - set(serveReqID))
                servedReqID = list(set(servedReqID).union(serveReqID))
                reward += len(serveReqID)
                # case 2:
                reward -= len(queueReqID)
                # case 3:
                retryFlag_reject = np.random.rand(len(rejectReqID)) < 0.7
                retryReqID = np.array(rejectReqID)[retryFlag_reject]
                cancelReqID = np.array(rejectReqID)[~retryFlag_reject]
                waitingReqID = list(set(waitingReqID) - set(rejectReqID))
                pendingReqID = list(set(pendingReqID).union(set(retryReqID)))
                failedReqID = list(set(failedReqID).union(set(cancelReqID)))
                reward -= len(retryReqID) + 10*len(cancelReqID)
                # case 4:
                reward -= len(unattendReqID)
                # case 5:
                if self.epoch == end_epoch:
                    reward -= 10*(len(pendingReqID) + len(waitingReqID))
                    failedReqID = list(set(failedReqID).union(pendingReqID).union(waitingReqID))
                    pendingReqID = []
                    waitingReqID = []

                # write-back dict
                pendingReqID_domain[domain] = pendingReqID
                waitingReqID_domain[domain] = waitingReqID
                servedReqID_domain[domain] = servedReqID
                failedReqID_domain[domain] = failedReqID
            # update active session buffer
            self.active_sessions.loc[sessionID, 'pendingReqID_per_domain'] = json.dumps(pendingReqID_domain)
            self.active_sessions.loc[sessionID, 'waitingReqID_per_domain'] = json.dumps(waitingReqID_domain)
            self.active_sessions.loc[sessionID, 'servedReqID_per_domain'] = json.dumps(servedReqID_domain)
            self.active_sessions.loc[sessionID, 'failedReqID_per_domain'] = json.dumps(failedReqID_domain)
        # Delete sessions that will end in next round from buffer =================================================
        left_next = self.head_datetime + (self.epoch+1)*self.time_step
        self.active_sessions.drop(
            self.active_sessions.index[self.active_sessions['endTime_datetime_updated'] < left_next],
            axis=0, inplace=True)
        # return reward ============================================================================================
        return reward
