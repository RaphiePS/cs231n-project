import numpy as np
import hyperparameters as hp
from action import Action
from collections import deque
import pdb

class Transition(object):
	def __init__(self, image, terminal, action, reward, was_start):
		# pdb.set_trace()
		self.action = action
		self.reward = reward
		self.image = image
		self.terminal = terminal
		self.was_start = was_start


class TransitionTable(object):

	def __init__(self):
		self.capacity = hp.REPLAY_MEMORY_SIZE
		self.transitions = deque(maxlen = self.capacity)

	def get_index(self, transition_index):
		s = np.zeros((hp.INPUT_SIZE, hp.INPUT_SIZE, hp.AGENT_HISTORY_LENGTH))
		sp = np.zeros((hp.INPUT_SIZE, hp.INPUT_SIZE, hp.AGENT_HISTORY_LENGTH))

		last_frame = self.transitions[transition_index]
		current_index = transition_index
		current_transition = self.transitions[current_index]
		t = current_transition.terminal
		a = current_transition.action.num
		r = current_transition.reward

		for i in range(hp.AGENT_HISTORY_LENGTH + 1):
			# print i, np.linalg.norm(current_transition.image)
			if i < hp.AGENT_HISTORY_LENGTH:
				sp[:, :, i] = current_transition.image
			if i > 0:
				s[:, :, i - 1] = current_transition.image
			if not current_transition.was_start:
				print "HELLO WE ARE DECREMENTING HERE"
				current_index -= 1
				current_transition = self.transitions[current_index]


		return (s, t, a, r, sp)

	def get_recent(self):
		# pick a new action
		s, t, a, r, sp = self.get_index(-1)
		return sp

	def get_minibatch(self):
		# gradient update
		size = hp.MINIBATCH_SIZE
		s = np.zeros((size, hp.INPUT_SIZE, hp.INPUT_SIZE, hp.AGENT_HISTORY_LENGTH))
		sp = np.zeros_like(s)
		t = np.zeros(size)
		a = np.zeros(size)
		r = np.zeros(size)

		for i in range(size):
			# index = np.random.randint(0, len(self.transitions))
			index = 1
			s[i], t[i], a[i], r[i], sp[i] = self.get_index(index)
			print s[i], t[i], a[i], r[i], sp[i]
		return s, t, a, r, sp

	def add_transition(self, image, terminal, action, reward, was_start):
		t = Transition(image, terminal, action, reward, was_start)
		self.transitions.append(t)





