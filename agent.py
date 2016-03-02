import numpy as np
from action import Action
import hyperparameters as hp
import tensorflow as tf
from double_qnet import update_target, minimize_loss, best_action, s, sp, actions, rewards, terminals
from transition_table import TransitionTable
import time

class Agent(object):
	def __init__(self):
		self.frame_count = -1
		self.sess = tf.Session()
		self.sess.run(tf.initialize_all_variables())
		self.transitions = TransitionTable()

	def epsilon(self):
		return ((hp.FINAL_EXPLORATION - hp.INITIAL_EXPLORATION) / hp.FINAL_EXPLORATION_FRAME) * self.frame_count + hp.INITIAL_EXPLORATION

	def step(self, image, reward, terminal, was_start, action):
		
		t = time.time()

		self.frame_count += 1

		self.transitions.add_transition(image, terminal, action, reward, was_start)
		if self.frame_count < hp.REPLAY_START_SIZE:
			return Action.random_action()

		if self.frame_count % hp.UPDATE_FREQUENCY == 0:
			
			s_, t_, a_, r_, sp_ = self.transitions.get_minibatch(self.frame_count)
			self.sess.run([minimize_loss], feed_dict={s: s_, sp: sp_, actions: a_, rewards: r_, terminals: t_})
			print "Update took %.2f" % ((time.time() - t) * 1000)

		if self.frame_count % (hp.UPDATE_FREQUENCY * hp.TARGET_UPDATE_FREQUENCY) == 0:
			self.sess.run(update_target)

		
		if np.random.rand() < self.epsilon():
			return Action.random_action()
		img = self.transitions.get_recent()
		return Action(self.sess.run(best_action[0], feed_dict={s: img}))