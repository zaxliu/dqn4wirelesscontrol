import numpy as np
import pandas as pd
import json

class TrafficEmulator(object):
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
                 rewarding=None, verbose=0):
        # Dataset =====================================================
        if session_df is None:
            raise ValueError("TrafficEmulator Initialization: session_df passed in is empty or None.")
        self.session_df = session_df  # Initialize session dataset
        # Time-related variables ======================================
        # starting datetime of the simulation, use datetime of first session arrival if not assigned
        self.head_datetime = head_datetime if head_datetime is not None \
            else self.session_df['startTime_datetime'].min()
        # end datetime of the simulation, use datetime of last session departure if not assigned
        self.tail_datetime = tail_datetime if tail_datetime is not None \
            else self.session_df['startTime_datetime'].max()
        if self.head_datetime > self.tail_datetime:  # sanity check
            raise ValueError("head_datetime > tail_datetime")
        self.time_step = time_step
        # Rewarding mechanism==========================================
        if rewarding is None:
            rewarding = {'serve': 1, 'wait': -1, 'fail': -10}
        self.Rs = rewarding['serve'] if 'serve' in rewarding else 1  # service reward, default 1
        self.Rw = rewarding['wait'] if 'wait' in rewarding else -1   # waiting reward, default -1
        self.Rf = rewarding['fail'] if 'fail' in rewarding else -10  # fail reward, default -10
        # Verbosity ===================================================
        self.verbose = verbose  # verbosity level
        # Reset active session buffer and time counter ================
        self.active_sessions = {}
        self.epoch = None
        self.reset()
        # Output ======================================================
        if verbose > 0:
            print " "*4 + "TrafficEmulator.__init__():",
            print "New TrafficEmulator with params:"
            print " "*8 + "head: {}, tail: {}, time_step: {}".format(
                self.head_datetime, self.tail_datetime, self.time_step
            )
            print " "*8 + "Rs: {}, Rw: {}, Rf: {}".format(
                self.Rs, self.Rw, self.Rf
            )

        return

    # Public Methods
    def generate_traffic(self):
        """Get traffic for this epoch, set epoch=None if run out of data
        First check the traffic dataset and add new sessions that are initiated during this epoch, add these new-coming
        sessions into the active session buffer. Then generate traffic (in term of how many bytes are sent in some
        request for some domain for each session).
        :return: traffic DataFrame
        """
        # 1. Add new incoming sessions to active session buffer, if already run out of data, suggest reset()
        if self.epoch is None:
            print "TrafficEmulator.__init__():",
            print "Run out of session data, please reset emulator!"
            return None

        left = self.head_datetime + self.epoch*self.time_step
        right = left + self.time_step

        num_drops = self.drop_expiring_sessions_(left)

        # if running out of data (left edge out of range), set epoch=None and return None
        if left >= self.tail_datetime:
            print "TrafficEmulator.generate_traffic():",
            print "Reach tail_datetime, please reset dataset!"
            self.epoch = None
            return None

        # else, get sessions initiated in this epoch
        num_incoming_sessions, num_active_sessions = self.append_to_active_sessions_(left, right)  # append incoming session to the active session buffer

        # logging
        if self.verbose > 0:
            print " "*4 + "TrafficEmulator.generate_traffic():",
            print "located {}, droped {}, left {} sessions."\
                  "".format(num_incoming_sessions,
                            num_drops,
                            num_active_sessions)
        traffic_df = self.generate_requests_()  # generate requests for current epoch
        return traffic_df

    def serve_and_reward(self, service_df):
        if self.epoch is not None:
            service_reward = self.evaluate_service_(service_df=service_df)  # update active session buffer and emit reward
            self.epoch += 1  # increase timer by one epoch
        else:
            print "TrafficEmulator.serve_and_reward():",
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
        self.active_sessions = {}
        # Time counter
        self.epoch = 0

    # Private Methods
    def append_to_active_sessions_(self, left, right):
        """Append incoming sessions to active session buffer and adjust format.

        :param incoming_sessions: incoming session DataFrame
        :return:
        """
        # Initialize internal state columns for each session
        # 'endTime_datetime_updated' column
        # incoming_sessions = self.session_df.loc[(self.session_df['startTime_datetime'] >= left) &
        #                                         (self.session_df['startTime_datetime'] < right)]
        left_idx = int(self.session_df['startTime_datetime'].searchsorted(left, side='left'))
        right_idx = int(self.session_df['startTime_datetime'].searchsorted(right, side='right'))
        incoming_sessions = self.session_df.iloc[left_idx:right_idx]
        incoming_sessions.loc[:, 'endTime_datetime_updated'] = incoming_sessions['endTime_datetime']
        for idx in incoming_sessions.index:
            # 'bytes_per_request_domain' column
            domains = incoming_sessions.get_value(idx, 'domains').split(';')
            bytes_domain = map(int, incoming_sessions.get_value(idx, 'bytesByDomain').split(';'))
            requests_domain = map(int, incoming_sessions.get_value(idx, 'requestsByDomain').split(';'))
            bytes_request_domain = {
                domains[i]: self.allocate_bytes_in_req(bytes_domain[i], requests_domain[i])
                for i in range(len(domains))
            }
            incoming_sessions.set_value(idx, 'bytes_per_request_domain', json.dumps(bytes_request_domain))
            start_epoch = int(np.floor((incoming_sessions.get_value(idx, 'startTime_datetime') - self.head_datetime) / self.time_step))
            end_epoch = int(np.floor((incoming_sessions.get_value(idx, 'endTime_datetime_updated') - self.head_datetime) / self.time_step))
            # 'pendingReqID_per_domain' column
            pendingReqID_epoch_domain = {
                domain: self.allocate_reqs_in_epoch(range(len(bytes_request_domain[domain])), start_epoch, end_epoch)
                for domain in bytes_request_domain
                }
            incoming_sessions.set_value(idx, 'pendingReqID_per_epoch_domain', json.dumps(pendingReqID_epoch_domain))

            # 'waitingReqID_per_domain' column
            incoming_sessions.set_value(idx, 'waitingReqID_per_domain', json.dumps({}))
            # 'servedReqID_per_domain' column
            incoming_sessions.set_value(idx, 'servedReqID_per_domain', json.dumps({}))
            # 'failedReqID_per_domain' column
            incoming_sessions.set_value(idx, 'failedReqID_per_domain', json.dumps({}))
        # Append incoming session to active session buffer
        if len(incoming_sessions) > 0:
            self.active_sessions[self.epoch] = incoming_sessions
        num_active_sessions = sum([len(sessions) for epoch, sessions in self.active_sessions.iteritems()])
        return len(incoming_sessions), num_active_sessions

    @staticmethod
    def allocate_bytes_in_req(bytes, requests):
        """ Distribute bytes among requests.

        Each byte independently goes into each request with equal probability. Assume B bytes and R
        requests, the bytes assigned to each request follow a multinomial distribution of B trails
        and R category.
        :param bytes: total bytes to assign
        :param requests: total requests to assign
        :return: a list of how many bytes per request
        """
        return list(np.random.multinomial(bytes, [1.0/requests]*requests))

    @staticmethod
    def allocate_reqs_in_epoch(reqID_list, start_epoch, end_epoch):
        epochs_total = end_epoch - start_epoch + 1
        bin_idx_per_req = np.floor(np.random.rand(len(reqID_list))*epochs_total)  # allocate reqs to virtual bins following certain distribution
        epochs = np.arange(start_epoch, end_epoch+1)  # shuffle epoch list to form virtual bins
        bin_idx_per_epoch = np.arange(len(epochs))
        np.random.shuffle(bin_idx_per_epoch)

        return {
            int(epoch): [
                reqID for (ir, reqID) in enumerate(reqID_list)
                if bin_idx_per_req[ir] == bin_idx_per_epoch[ie]
                ]
            for (ie, epoch) in enumerate(epochs)
            if sum(bin_idx_per_req == bin_idx_per_epoch[ie]) > 0
            }

    def generate_requests_(self):
        """How many requests in each domain are issued in current epoch?

        Traffic drilling-down assumptions:
            Assume traffics from each session are generated independently. Each pending request is sent independently in
            the current epoch with probablity 1/epochs_left, i.e. uniformly distributed in the expected future.
        :return: a DataFrame with traffic info
        """
        traffic_df = pd.DataFrame(data=None, index=None, columns=['sessionID', 'uid', 'bytesSent_per_request_domain'])
        num_req = 0

        # Generate requests from each session in the active session buffer
        for epoch_key, sessions in self.active_sessions.iteritems():
            for sessionID in sessions.index:
                # extract info from active session buffer
                bytes_req_domain = json.loads(sessions.get_value(sessionID, 'bytes_per_request_domain'))
                pendingReqID_epoch_domain = json.loads(sessions.get_value(sessionID, 'pendingReqID_per_epoch_domain'))
                waitingReqID_domain = json.loads(sessions.get_value(sessionID, 'waitingReqID_per_domain'))

                # Which requests are sent in current epoch under each domain and how many bytes?
                bytesSent_req_domain = {}  # {domain1: [req1_id: req1_bytes, req2_id: req2_bytes, ...], domain2: ...}
                for domain in bytes_req_domain:
                    # extract info lists from dicts
                    pendingReqID_epoch = pendingReqID_epoch_domain[domain] if domain in pendingReqID_epoch_domain else {}
                    waitingReqID = waitingReqID_domain[domain] if domain in waitingReqID_domain else []

                    # decide which req to sent and update info
                    if str(self.epoch) in pendingReqID_epoch:
                        toSendReqID = pendingReqID_epoch[str(self.epoch)]
                        del pendingReqID_epoch[str(self.epoch)]
                    else:
                        toSendReqID = []
                    pendingReqID_epoch_domain[domain] = pendingReqID_epoch
                    waitingReqID = list(set(waitingReqID).union(set(toSendReqID)))  # add toSendReq into waiting list
                    waitingReqID_domain[domain] = waitingReqID

                    # construct traffic dict
                    if len(toSendReqID) > 0:  # build traffic dict
                        bytesSent_req_domain[domain] = {ID: bytes_req_domain[domain][ID] for ID in toSendReqID}
                    num_req += len(toSendReqID)

                # update active session buffer
                sessions.set_value(sessionID, 'pendingReqID_per_epoch_domain', json.dumps(pendingReqID_epoch_domain))
                sessions.set_value(sessionID, 'waitingReqID_per_domain', json.dumps(waitingReqID_domain))

                # generate current_traffic
                if len(bytesSent_req_domain) > 0:
                    traffic_df = traffic_df.append(pd.DataFrame(
                        {'sessionID': sessionID,
                         'uid': sessions.get_value(sessionID, 'uid'),
                         'bytesSent_per_request_domain': json.dumps(bytesSent_req_domain)}, index=[None]),
                        ignore_index=True)

        if self.verbose > 1:
            print " "*8 + "TrafficEmulator.generate_requests_():",
            print "generated {} requests.".format(num_req)

        return traffic_df

    def evaluate_service_(self, service_df):
        """Modify active session buffer according to the service provided and emit reward

        Interaction assumptions:
            1: if a "waiting" request is served in the current epoch, emit +1 and set state as "served".
            2. if this is the last epoch, then assume all "waiting" and "pending" requests "failed", emit -10.
            3: if a "waiting" request is queued in the current epoch, emit -1 and keep the state as "waiting".
            4. if a "waiting" request is rejected in the current epoch, then emit -1 and set state as "pending" with
               probability 0.7 or emit -10 and set state as "failed" with probability 0.3. (0.7-persistent retransmission)
            5. if no instruction for a request, assume the server means to "queue" it, emit -1.

        :param service_df: a data frame indicating the service for each traffic row
        :return: a scalar last_reward for the service
        """
        reward_s = 0  # Set initial rewards to zero
        reward_w = 0
        reward_f = 0
        num_serving_c = 0
        num_queuing_c = 0
        num_rejecting_c = 0
        num_retried_c = 0
        num_canceled_c = 0
        num_unattended_c = 0
        num_pending = 0
        num_waiting = 0
        num_served = 0
        num_failed = 0

        for epoch_key, sessions in self.active_sessions.iteritems():
            for sessionID in sessions.index:  # interacte with each service row in service_df
                # extract service info from service_df
                idx = (service_df['sessionID'] == sessionID).nonzero()[0] \
                    if len(service_df) > 0 else np.array([])
                if len(idx) == 1:
                    service_req_domain = json.loads(service_df.get_value(idx[0], 'service_per_request_domain'))
                elif len(idx) == 0:
                    service_req_domain = {}
                else:
                    raise ValueError("TrafficEmulator.evaluate_service_(): more than 1 service entry for a session")

                # extract info from active session buffer
                end_epoch = int(np.floor(
                    (sessions.get_value(sessionID, 'endTime_datetime_updated') - self.head_datetime) / self.time_step))
                pendingReqID_epoch_domain = json.loads(sessions.get_value(sessionID, 'pendingReqID_per_epoch_domain'))
                waitingReqID_domain = json.loads(sessions.get_value(sessionID, 'waitingReqID_per_domain'))
                servedReqID_domain = json.loads(sessions.get_value(sessionID, 'servedReqID_per_domain'))
                failedReqID_domain = json.loads(sessions.get_value(sessionID, 'failedReqID_per_domain'))

                # for each domain, check the services that 'waiting' requests received, and update active session
                # buffer accordingly
                for domain in waitingReqID_domain:
                    # extract info from dicts
                    service_req = service_req_domain[domain] if domain in service_req_domain else {}
                    pendingReqID_epoch = pendingReqID_epoch_domain[domain] if domain in pendingReqID_epoch_domain else {}
                    waitingReqID = waitingReqID_domain[domain] if domain in waitingReqID_domain else []
                    servedReqID = servedReqID_domain[domain] if domain in servedReqID_domain else []
                    failedReqID = failedReqID_domain[domain] if domain in failedReqID_domain else []
                    # indices of requests which are served, queued, and rejected...
                    # Note: json converts key to string type
                    servingReqID = [int(req_id) for req_id in service_req if service_req[req_id] == 'serve']
                    queuingReqID = [int(req_id) for req_id in service_req if service_req[req_id] == 'queue']
                    rejectingReqID = [int(req_id) for req_id in service_req if service_req[req_id] == 'reject']
                    # counters
                    num_serving_c += len(servingReqID)
                    num_queuing_c += len(queuingReqID)
                    num_rejecting_c += len(rejectingReqID)

                    # Case 1: served requests
                    reward_s += self.Rs*len(servingReqID)
                    waitingReqID = list(set(waitingReqID) - set(servingReqID))  # move req from waiting list to servied list
                    servedReqID = list(set(servedReqID).union(servingReqID))
                    # Case 2: end of session
                    if self.epoch == end_epoch:
                        reward_f += self.Rf*(reduce(lambda x, y: x+len(y), pendingReqID_epoch.values(), 0) + len(waitingReqID))
                        failedReqID = list(set(failedReqID).union(
                            reduce(lambda x,y: x+y, pendingReqID_epoch.values(), [])
                        ).union(waitingReqID))
                        pendingReqID_epoch = {}  # pending req ID is a dict
                        waitingReqID = []
                    else:
                        # Case 3: queued requests
                        reward_w += self.Rw*len(queuingReqID)
                        # Case 4: rejected requests
                        retryFlag = np.random.rand(len(rejectingReqID)) < 0.5
                        retryReqID = np.array(rejectingReqID)[retryFlag]
                        cancelReqID = np.array(rejectingReqID)[~retryFlag]
                        waitingReqID = list(set(waitingReqID) - set(rejectingReqID))
                        if len(retryReqID) > 0:  # the following fcn call is expensive even if retryReqID is empty
                            pendingReqID_epoch.update(self.allocate_reqs_in_epoch(retryReqID, self.epoch+1, end_epoch))
                        failedReqID = list(set(failedReqID).union(set(cancelReqID)))
                        reward_w += self.Rw*len(retryReqID)
                        reward_f += self.Rf*len(cancelReqID)
                        num_retried_c += sum(retryFlag)
                        num_canceled_c += sum(~retryFlag)
                        # case 5: unattended requests
                        unattendingReqID = list(set(waitingReqID)-set(queuingReqID))
                        reward_w += self.Rw*len(unattendingReqID)
                        num_unattended_c += len(unattendingReqID)

                    # write-back dict
                    pendingReqID_epoch_domain[domain] = pendingReqID_epoch
                    waitingReqID_domain[domain] = waitingReqID
                    servedReqID_domain[domain] = servedReqID
                    failedReqID_domain[domain] = failedReqID

                    # counters
                    num_pending += reduce(lambda x, y: x+len(y), pendingReqID_epoch.values(), 0)
                    num_waiting += len(waitingReqID)
                    num_served += len(servedReqID)
                    num_failed += len(failedReqID)

                # update active session buffer
                sessions.set_value(sessionID, 'pendingReqID_per_epoch_domain', json.dumps(pendingReqID_epoch_domain))
                sessions.set_value(sessionID, 'waitingReqID_per_domain', json.dumps(waitingReqID_domain))
                sessions.set_value(sessionID, 'servedReqID_per_domain', json.dumps(servedReqID_domain))
                sessions.set_value(sessionID, 'failedReqID_per_domain', json.dumps(failedReqID_domain))

        if self.verbose > 1:
            print " "*8 + "TrafficEmulator.evaluate_service_():",
            print "served {}, queued {}, rejected {} ({}, {}), " \
                  "unattended {}, reward {} ({}, {}, {})".format(
                      num_serving_c, num_queuing_c,
                      num_rejecting_c, num_retried_c, num_canceled_c,
                      num_unattended_c,
                      reward_s+reward_w+reward_f,
                      reward_s, reward_w, reward_f)
            print " "*8 + "TrafficEmulator.evaluate_service_():",
            print "pending {}, waiting {}, served {}, failed {}".format(
                num_pending, num_waiting, num_served, num_failed)

        return reward_s+reward_w+reward_f

    def drop_expiring_sessions_(self, left):
        num_drops = 0
        drop_df_keys = []
        for epoch_key, sessions in self.active_sessions.iteritems():
            drop_index = sessions.index[sessions['endTime_datetime_updated'] < left]
            sessions.drop(drop_index, axis=0, inplace=True)
            num_drops += len(drop_index)
            drop_df_keys.append(epoch_key) if len(sessions) == 0 else None
        for epoch_key in drop_df_keys:
            del self.active_sessions[epoch_key]
        return num_drops


class PoissonEmulator(TrafficEmulator):
    def __init__(self, mu=1, **kwargs):
        super(PoissonEmulator, self).__init__(**kwargs)
        self.MU = mu

    def append_to_active_sessions_(self, left, right):
        """Generate poisson traffic and append to

        """
        incoming_sessions = pd.DataFrame(data=[],
                                         index=[self.epoch],
                                         columns=['uid',
                                                  'bytes_per_request_domain',
                                                  'pendingReqID_per_epoch_domain',
                                                  'waitingReqID_per_domain',
                                                  'servedReqID_per_domain',
                                                  'failedReqID_per_domain'])
        incoming_sessions.set_value(self.epoch, 'uid', 0)
        incoming_sessions.set_value(self.epoch, 'endTime_datetime_updated', self.tail_datetime)

        idx = self.epoch
        # 'bytes_per_request_domain' column
        n_requests = self.emit_requests_()
        bytes_request_domain = {'fake.com': [-1]*n_requests}
        incoming_sessions.set_value(idx, 'bytes_per_request_domain', json.dumps(bytes_request_domain))
        # 'pendingReqID_per_domain' column
        pendingReqID_epoch_domain = {'fake.com': {self.epoch: range(n_requests)}}
        incoming_sessions.set_value(idx, 'pendingReqID_per_epoch_domain', json.dumps(pendingReqID_epoch_domain))

        # 'waitingReqID_per_domain' column
        incoming_sessions.set_value(idx, 'waitingReqID_per_domain', json.dumps({}))
        # 'servedReqID_per_domain' column
        incoming_sessions.set_value(idx, 'servedReqID_per_domain', json.dumps({}))
        # 'failedReqID_per_domain' column
        incoming_sessions.set_value(idx, 'failedReqID_per_domain', json.dumps({}))

        # Append incoming session to active session buffer
        if len(incoming_sessions) > 0:
            self.active_sessions[self.epoch] = incoming_sessions
        num_active_sessions = sum([len(sessions) for epoch, sessions in self.active_sessions.iteritems()])
        return len(incoming_sessions), num_active_sessions

    def drop_expiring_sessions_(self, left):
        num_drops = 0
        drop_df_keys = []
        for epoch_key, sessions in self.active_sessions.iteritems():
            drop_index = [sessionID for sessionID in sessions.index
                          if PoissonEmulator.if_session_empty(sessionID, sessions)]
            sessions.drop(drop_index, axis=0, inplace=True)
            num_drops += len(drop_index)
            drop_df_keys.append(epoch_key) if len(sessions) == 0 else None
        for epoch_key in drop_df_keys:
            del self.active_sessions[epoch_key]
        return num_drops

    @staticmethod
    def if_session_empty(sessionID, sessions):
        pendingReqID_epoch_domain = json.loads(sessions.get_value(sessionID, 'pendingReqID_per_epoch_domain'))
        waitingReqID_domain = json.loads(sessions.get_value(sessionID, 'waitingReqID_per_domain'))
        num_pending = sum([len(pendingReqID_epoch_domain[domain][epoch])
                           for domain in pendingReqID_epoch_domain
                           for epoch in pendingReqID_epoch_domain[domain]])
        num_waiting = sum([len(waitingReqID_domain[domain]) for domain in waitingReqID_domain])
        return num_pending==0 and num_waiting==0

    def emit_requests_(self):
        return np.random.poisson(self.MU)


class MMPPEmulator(PoissonEmulator):
    def __init__(self, traffic_model, **kwargs):
        super(MMPPEmulator, self).__init__(**kwargs)
        self.traffic_model = traffic_model  # assume model properly initialized
        self.BUFFER_SIZE = 1000
        self.buffer = None
        self.counter = 0

    def emit_requests_(self):
        if self.buffer is None or self.counter >= self.BUFFER_SIZE:
            X, Z = self.traffic_model.sample(self.BUFFER_SIZE)
            self.traffic_model.startprob_ *= 0.0
            self.traffic_model.startprob_[Z[-1]] = 1.0
            self.buffer = X.squeeze()
            self.counter = 0
        n_requests = self.buffer[self.counter]
        self.counter += 1
        
        return n_requests



