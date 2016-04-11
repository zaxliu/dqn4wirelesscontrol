from qlearning.qtable import QAgent as Agent
import numpy as np


class BaseNode(object):
    def __init__(self, id=0, agent=None, traffic=None, routing_childs=None, parent = 0, epoch=0, queue=0, offset=0, chanLen=0, histLen =1):
        self.id = id
        self.agent = Agent() if agent is None else agent  # some intelligent agent with a learning and decision interface
        self.traffic = traffic  # responsible for generating traffic
        self.routing_childs = routing_childs  # who to receive
        self.parent = parent #parent id
        self.epoch = epoch # sending period
        self.counter= epoch # time counter
        self.queue = queue # queue info
        self.offset = offset # starting time to send a packet
        #self.channelLen = [None] * channelLen # a list of actions on each channel
        self.chanLen = chanLen # num of channels
        self.observation = [None]*histLen # past ACTIONS + traffic info + etc...
        

    
#=====================================================
    def step(self, last_ack):
        #send packet
        self.send_packet_()

          
        reward = self.evaluate_ack_(last_ack)
        action, _ = self.agent.observe_and_act(observation=np.array(self.observation), last_reward=reward)
        self.observation.append(action)

        if self.observation[-1] is not None:
            self.observation.pop(0)

        """Example of action:
           [channelIndex, act]
            1. channelIndex: Index of channel; Note: channel index starts from 0: 0,1,2,...
            2. act: 
                1: send
                0: idle
                -1: receive
        """
        #enqueue or dequeue according to last_ack
        if last_ack is not None:
            temp = self.observation(len(self.observation))
            if temp[1] == 1:
                self.queue -= 1 if self.queue > 0 else 0
            if temp[1] == -1:
                self.queue = self.queue+1

        return self.translate_action_(action if self.queue > 0 else (0, 0)), reward

#======================================================

    def reset(self):
        pass

#======================================================
    def evaluate_ack_(self, ack):
        """Evaluate last TRX and emit reward
        """

        if ack is None:
            return 0
        else:
            if ack in self.routing_childs:
                return 1
            else:
                return 0
            
#=====================================================    

    def translate_action_(self, action):
        """translate agent's action into a transmit action

        Example:
            action description
                1. None: nothing to do
                2. (+1, sender_id, receiver_id), tuple: send to receiver on this channel
                3. (-1, ) tuple:  listen on this channel

            example return 1) [None, None, (-1, ), None] 2) [None, None, (+1, self.id, 3), None]

        :param action: pass
        :return: a list of actions on each channel
        """
        # translate action in to trx cmd
        trx = [None]*self.chanLen
        #action: [channel, act]
        if action[1] == 1:
            trx[action[0]] = (action[1], self.id, self.parent)
        elif action[1] == -1:
            trx[action[0]] = (-1, 0, 0)
        else:
            pass
        return trx


    #===============================================
    def send_packet_(self):
        if self.offset > 0:
            self.offset = self.offset-1

        else:
            self.counter=self.counter-1

        if self.counter == 0:
            self.queue=self.queue+1
            self.counter = self.epoch









