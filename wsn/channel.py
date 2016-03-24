class BaseChannel(object):
    """Base class for channel implemetations

    """
    def __init__(self):
        self.PTR = None  # summarize channel condition in a success rate matrix

    def communicate(self, transmission_and_reception):
        # decide whether each transmission or reception is successful
        # fill up an acknowlegement for each node
        acknowlegement = None
        return acknowlegement