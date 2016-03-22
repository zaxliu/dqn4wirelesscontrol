from collections import deque
import numpy as np


class PhiMixin(object):
    """Phi function buffers the past PHI_LENGTH (action, observation) pairs to form an agent state.
    Currently only support None, 1-d, and 2-d observations and scalar action index. The action index is one-hot encoded
    as a vector to append to the last dimension of observation. It is repeated along other dimensions to match the shape
    of observations.

    Note: this is a Mixin class, meaning some attributes or methods used here is not initialized and need to be
    implemented somewhere else in the inheritance hierarchy.
    """
    def __init__(self, phi_length, **kwargs):
        self.PHI_LENGTH = phi_length
        self.phi_buffer = deque()
        super(PhiMixin, self).__init__(**kwargs)  # pass on key-word arguments for initialization of parent classes

    def transition_(self, observation, reward):
        """The Phi function
        This is where other classes make function calls. In the end, escalate function call to the transition_() method
        of the parent class to act like a "decorator".

        Parameters
        ----------
        observation : Convertable to numpy array. 1d, 2d, or None.
        reward :

        Returns : the current agent state
        -------

        """
        last_action = self.last_action
        if last_action is None:  # if no action was taken, return None state
            return None
        idx_action = self.ACTIONS.index(last_action)

        # Update buffer
        state_slice = self.combine_(observation, idx_action)  # get current slice of agent state
        self.phi_buffer.append(state_slice)
        if len(self.phi_buffer) > self.PHI_LENGTH:  # maintain length
            self.phi_buffer.popleft()

        # Return the whole state tensor
        state = np.array(self.phi_buffer)
        if len(self.phi_buffer) < self.PHI_LENGTH:  # return None if buffer is still filling up
            return None
        else:
            try:  # try escalate call to parent classes
                state = super(PhiMixin, self).transition_(observation=state, reward=reward)
            except AttributeError:
                pass
            finally:
                return state

    def combine_(self, observation, idx_action):
        """Combine observation and action index into a single tensor

        Parameters
        ----------
        observation : None, 1d, or 2d
        idx_action : scalar
        """
        ob = np.array(observation)
        if len(ob.shape) == 1:
            ac_shape = (len(self.ACTIONS),)
            ac = np.zeros(shape=ac_shape)
            ac[idx_action] = 1
        elif len(ob.shape) == 2:
            ac_shape = (ob.shape[0], len(self.ACTIONS))
            ac = np.zeros(shape=ac_shape)
            ac[:, idx_action] = 1
        else:
            raise ValueError("Unsupported observation shape {}".format(ob.shape))
        state = np.concatenate([ob, ac])
        return state