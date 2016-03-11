import numpy as np
import hyperparameters as hp

class ActionMeta(type):
    def __init__(cls, name, bases, d):
        type.__init__(cls, name, bases, d)
        cls.action_to_num = dict()
        cls.num_to_action = dict()
        options = [True, False]
        counter = 0
        for left in options:
            for right in options:
                for faster in options:
                    for slower in options:
                        if left and right:
                            continue
                        # if faster and slower:
                        if slower:
                            continue
                        if not faster:
                            continue
                        a = (left, right, faster, slower)
                        cls.action_to_num[a] = counter
                        cls.num_to_action[counter] = a
                        counter += 1
        cls.num_actions = counter

class Action(object):
    __metaclass__ = ActionMeta
    
    def __init__(self, num=None, left=False, right=False, faster=False, slower=False):
        if num is not None:
            if num > hp.TOTAL_ACTIONS-1 or num < 0:
                raise ValueError("Invalid num, must be 0-5")
            self.num = num
        else:
            if left and right:
                raise ValueError("Invalid action, cannot press both left and right")
            if faster and slower:
                raise ValueError("Invalid action, cannot press both faster and slower")
            self.num = self.action_to_num[(left, right, faster, slower)]
    
    @classmethod
    def random_action(self):
        return Action(np.random.randint(0, self.num_actions))

    def to_onehot(self):
        a = np.zeros(self.num_actions)
        a[self.num] = 1
        return a

    def to_dict(self):
        left, right, faster, slower = self.num_to_action[self.num]
        return {
            "keyLeft": left,
            "keyRight": right,
            "keyFaster": faster,
            "keySlower": slower
        }