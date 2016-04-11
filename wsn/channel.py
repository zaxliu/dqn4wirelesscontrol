class BaseChannel(object):

    """Base class for channel implemetations

    """

    def __init__(self):
        self.PTR = None  # summarize channel condition in a success rate matrix

    def communicate(self, transmission_and_reception):
        """Process each node's transmission action and return nodes' ACK

        Decide whether each transmission or reception is successful, and
        fill up an acknowlegement for each node.

        ACK explanation: a node's ACK is int or None
            - if the node gets no response (to send or receive or do nothing), 
                the value is None.
            - otherwise, the value is the respsonser id. For sender, it's the
                receiver's id and for receiver, it's sender's id.

        :param tranmission_and_reception: list of nodes' tranmission actions
        :return: list of nodes' ACKs
        """

        nodes_num = len(transmission_and_reception)
        records = [None for i in range(nodes_num)]
        acknowlegement = [None for i in range(nodes_num)]

        channels_num = len(transmission_and_reception[0])
        channels = [{
            'senders': [],
            'receivers': []
        } for i in range(channels_num)]


        for j in range(nodes_num):
            node_action = transmission_and_reception[j]
            for i in range(channels_num):
                if node_action[i] == None:
                    continue
                elif node_action[i][0] == -1:
                    channels[i]['receivers'].append(j)
                    records[i] = (-1, [])
                elif node_action[i][0] == 1:
                    channels[i]['senders'].append(node_action[i][1:])
                    records[i] = (1, [])

        for i in range(channels_num):
            senders = channels[i]['senders']
            receivers = channels[i]['receivers']
            for sender in senders:
                if sender[1] in receivers:
                    records[sender[1]][1].append(sender[0])

        for i in range(nodes_num):
            record = records[i]
            if record is not None and record[0] == -1:
                if len(record[1]) == 1:
                    acknowlegement[i] = record[1][0]
                    acknowlegement[record[1][0]] = i

        return acknowlegement
