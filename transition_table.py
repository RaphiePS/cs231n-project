import numpy as np
import hyperparameters as hp
from action import Action
from collections import deque
import pdb

class Transition(object):
	def __init__(self, image, terminal, action, reward, was_start, telemetry):
		self.action = action
		self.reward = reward
		self.image = image
		self.terminal = terminal
		self.was_start = was_start
		self.telemetry = telemetry


class TransitionTable(object):

	def __init__(self):
		self.capacity = hp.REPLAY_MEMORY_SIZE
		self.transitions = deque(maxlen = self.capacity)

	def get_index(self, transition_index):
		s = np.zeros((hp.INPUT_SIZE, hp.INPUT_SIZE, hp.NUM_CHANNELS))
		sp = np.zeros((hp.INPUT_SIZE, hp.INPUT_SIZE, hp.NUM_CHANNELS))

		last_frame = self.transitions[transition_index]
		current_index = transition_index
		current_transition = self.transitions[current_index]
		t = int(current_transition.terminal)
		a = current_transition.action.num
		r = current_transition.reward

		sp = current_transition.image
		if not current_transition.was_start:
			s = self.transitions[current_index - 1].image
		else:
			s = current_transition.image

		# for i in range(hp.AGENT_HISTORY_LENGTH + 1):
		# 	if i < hp.AGENT_HISTORY_LENGTH:
		# 		sp[:, :, :, i] = current_transition.image
		# 	if i > 0:
		# 		s[:, :, :, i - 1] = current_transition.image
		# 	if not current_transition.was_start:
		# 		current_index -= 1
		# 		current_transition = self.transitions[current_index]

		return (s, t, a, r, sp)

	def get_recent(self):
		# pick a new action
		s, t, a, r, sp = self.get_index(-1)
		return sp.reshape((1, hp.INPUT_SIZE, hp.INPUT_SIZE, hp.NUM_CHANNELS))

	def get_minibatch(self, frame_count):
		# gradient update
		size = hp.MINIBATCH_SIZE
		s = np.zeros((size, hp.INPUT_SIZE, hp.INPUT_SIZE, hp.NUM_CHANNELS))
		sp = np.zeros_like(s)
		t = np.zeros(size)
		a = np.zeros(size)
		r = np.zeros(size)

		for i in range(size):
			lower = 1 if frame_count >= self.capacity else 0
			index = np.random.randint(lower, len(self.transitions))
			s[i], t[i], a[i], r[i], sp[i] = self.get_index(index)
		return s, t, a, r, sp

	def add_transition(self, image, terminal, action, reward, was_start, telemetry):
		t = Transition(image, terminal, action, reward, was_start, telemetry)
		self.transitions.append(t)
