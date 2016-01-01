import numpy as np
import pandas as pd
import json


class TrafficEmulator:
    """Emulate traffic interaction between users and server

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
    def __init__(self, session_df, head_datetime=None, tail_datetime=None, time_step=pd.Timedelta(seconds=1),
                 verbose=0):
        self.session_df = session_df  # Initialize session dataset
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
        self.epoch = 0
        self.active_sessions = pd.DataFrame(columns=self.session_df.columns | pd.Index([
            'endTime_datetime_updated', 'bytesLeft_per_request_per_domain', 'requestsSent_per_domain'
        ]))  # buffer for active sessions
        self.verbose = verbose  # verbosity level

    # Public Methods
    def get_traffic(self):
        """Get traffic for this epoch, set epoch=None if run out of data

        :return: traffic DataFrame
        """
        # Add new incoming sessions to active session buffer
        # if already run out of data, suggest reset()
        if self.epoch is None:
            print "Run out of session data, please reset emulator."
            return None
        # get left and right edge of current epoch
        left = self.head_datetime + self.epoch*self.time_step
        right = left + self.time_step
        # if left edge out of range, set epoch=None to indicate run out of data
        if left >= self.tail_datetime:
            print "Run out of session data, please reset dataset"
            self.epoch = None
            return None
        incoming_sessions = self.session_df.loc[(self.session_df['startTime_datetime'] >= left) &
                                                (self.session_df['startTime_datetime'] < right)]
        self.__append_to_buffer__(incoming_sessions)
        # Generate traffic according active session buffer content
        traffic_df = self.__generate_traffic__()
        return traffic_df

    def serve(self, service):
        # Update active session buffer according to traffic
        service_reward = self.__update_buffer__(service=service)
        # Delete sessions that will end in next round
        left_next = self.head_datetime + (self.epoch+1)*self.time_step
        self.active_sessions.drop(
            self.active_sessions.index[self.active_sessions['endTime_datetime_updated'] < left_next],
            axis=0, inplace=True)
        # Increase timer by one epoch
        self.epoch += 1
        return service_reward

    def reset(self):
        self.epoch = 0  # Reset epoch to 0
        self.active_sessions = pd.DataFrame()  # Clear active session buffer

    # Private Methods
    def __append_to_buffer__(self, incoming_sessions):
        """Append incoming sessions to active session buffer and adjust format.
        :param incoming_sessions: incoming session DataFrame
        :return:
        """
        # Initialize internal state columns for each session
        # when is the session expected to end? initialize with original end time, may be modified based on service
        incoming_sessions.loc[:, 'endTime_datetime_updated'] = incoming_sessions['endTime_datetime']
        # how many bytes goes into each requests of each domain?
        # Stored as json-encoded dict, key is domain, value is bytes per request list
        for idx in incoming_sessions.index:
            domains = incoming_sessions.loc[idx, 'domains'].split(';')
            bytes_domain = map(int, incoming_sessions.loc[idx, 'bytesByDomain'].split(';'))
            requests_domain = map(int, incoming_sessions.loc[idx, 'requestsByDomain'].split(';'))
            incoming_sessions.loc[idx, 'bytesLeft_per_request_per_domain'] = json.dumps(dict(zip(
                domains, [self.__distribute_bytes__(bytes_domain[i], requests_domain[i]) for i in range(len(domains))])))
        # which requests of each domain are sent in the last epoch (json-encoded dict)
        # Append incoming session to active session buffer
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

    def __generate_traffic__(self):
        """How many requests in each domain are issued in current epoch?
        Assume traffics from each session are generated independently. Each request are sent independently in
        the current epoch with probablity 1/epochs_left, i.e. uniformly distributed expected future epochs.
        :return: a DataFrame with traffic info
        """
        traffic_df = pd.DataFrame(data=None, index=None, columns=['uid', 'bytesSent_per_request_per_domain'])
        # process each session independently
        for idx in self.active_sessions.index:
            # how many epochs left for the current session?
            end_epoch = np.floor((self.active_sessions.loc[idx, 'endTime_datetime_updated'] - self.head_datetime
                                  ) / self.time_step)
            epochs_left = end_epoch - self.epoch + 1
            bytesLeft_req_domain = json.loads(self.active_sessions.loc[idx, 'bytesLeft_per_request_per_domain'])
            reqSent_domain = {}
            bytesSent_req_domain = {}
            # Which reqs  are sent for each domain?
            for domain in bytesLeft_req_domain:
                flag = np.random.rand(1, len(bytesLeft_req_domain[domain])) < 1.0/epochs_left  # send or not?
                reqSent_domain[domain] = flag.nonzero()[1].tolist()  # record the index of sent requensts
                bytesSent_req_domain[domain] = np.array(bytesLeft_req_domain[domain])[flag.nonzero()[1]].tolist()  # generate traffic for sent requests
            rsd_str = json.dumps(reqSent_domain)
            bsrd_str = json.dumps(bytesSent_req_domain)
            self.active_sessions.loc[idx, 'requestsSent_per_domain'] = rsd_str
            traffic_df = traffic_df.append(pd.DataFrame({'uid': self.active_sessions.loc[idx, 'uid'],
                                                         'bytesSent_per_request_per_domain': bsrd_str}, index=[idx]))
        return traffic_df

    def __update_buffer__(self, service):
        """Modify active session state according to the service provided
        Interaction assumptions:
            1: if a request is served, then substract this request from the buffer.
            2: if a request is denied, then keep it in the buffer for next round.
        Reward assumptions:
            1: if a request is served, emit 1
            2: if a request is denied, emit -1
        :param service: a data frame indicating the service for each traffic row
        :return: a scalar reward for the service
        """
        reward = 0
        for idx in service.index:
            bytesLeft_req_domain = json.loads(self.active_sessions.loc[idx, 'bytesLeft_per_request_per_domain'])
            reqSent_domain = json.loads(self.active_sessions.loc[idx, 'requestsSent_per_domain'])
            reqServedFlag_domain = json.loads(service.loc[idx, 'reqServedFlag_per_domain'])
            for domain in reqSent_domain:
                bytesLeft_req = np.array(bytesLeft_req_domain[domain])
                reqSent = np.array(reqSent_domain[domain])
                servedFlag = np.array(reqServedFlag_domain[domain])  # a list of if-served flag for each req sent
                leftFlag = [i not in reqSent[servedFlag].tolist() for i in range(len(bytesLeft_req))]
                bytesLeft_req = bytesLeft_req[leftFlag]  # delete the served requests
                bytesLeft_req_domain[domain] = bytesLeft_req.tolist()
                self.active_sessions.loc[idx, 'bytesLeft_per_request_per_domain'] = json.dumps(bytesLeft_req_domain)
                reward += (np.array(servedFlag)*2-1).sum()  # accumulate rewards for current domain
        return reward

