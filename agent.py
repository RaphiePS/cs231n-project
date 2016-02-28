import numpy as np
from action import Action

def step(image, reward, terminal):
	return Action(left=False, right=False, faster=True, slower=False)