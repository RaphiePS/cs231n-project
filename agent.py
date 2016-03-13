import numpy as np
from action import Action
import hyperparameters as hp
import tensorflow as tf
from double_qnet import update_target, minimize_loss, loss, best_action, s, sp, actions, rewards, terminals, all_vars, avg_q_val
from transition_table import TransitionTable
import time
import pickle
import sys
from sys import platform as _platform

class Agent(object):
	def __init__(self):
		self.frame_count = -1
		self.sess = tf.Session()
		self.transitions = TransitionTable()
		self.tfSaver = tf.train.Saver()
		self.timer = dict()
		self.timer[-1] = time.time()
		self.test_mode = False

		if len(sys.argv) > 2:
			if sys.argv[2] == "--run":
				path = sys.argv[1]
				self.tfSaver.restore(self.sess, path)
				print "RESTORED MODEL FROM PATH: %s" % path
				self.test_mode = True
			elif sys.argv[2] == "--train":
				path = sys.argv[1]
				f = open(path, 'r')
				to_restore = pickle.load(f)
				f.close()
				self.frame_count, self.transitions = to_restore['frame_count'], to_restore['transitions']
				print "RESTORED TRANSITIONS FROM PATH %s to FRAME COUNT %d" % (path, self.frame_count)
				self.sess.run(tf.initialize_all_variables())
			else:
				path = sys.argv[1]
				self.tfSaver.restore(self.sess, path)
				print "RESTORED MODEL FROM PATH: %s" % path
				f = open(sys.argv[2], 'r')
				to_restore = pickle.load(f)
				f.close()
				self.frame_count, self.transitions = to_restore['frame_count'], to_restore['transitions']
				print "RESTORED TRANSITIONS FROM PATH %s to FRAME COUNT %d" % (path, self.frame_count)
		else:
			self.sess.run(tf.initialize_all_variables())

	def in_lane(self, pos):
		pos = abs(pos)
		return pos <= 0.2 or (pos >= 0.5 and pos <= 0.8)

	def frame_reward(self, frame):

		if hp.REWARD_FUNC == 'raphie':
			if frame['collision']:
				return -1.0
			elif abs(frame['position']) > 0.8:
				return -0.8
			elif float(frame['speed']) == 0:
				return -1.0
			else:
				multiplier = 1.0 if self.in_lane(frame['position']) else 0.8
				return multiplier * (float(frame['speed']) / float(frame['max_speed']))
		
		elif hp.REWARD_FUNC == 'rishi':
			if frame['collision'] or abs(frame['position']) > 0.8:
				return -5.0
			elif frame['speed'] == 0:
				return -10.0
			else:
				return min(10, .2 + (5 * float(frame['speed']) / float(frame['max_speed'])))

		elif hp.REWARD_FUNC == 'position':
			if frame['collision']:
				return -1.0
			elif abs(frame['position']) > 0.8:
				return -0.8
			else:
				multiplier = 3.0 if self.in_lane(frame['position']) else 1.0
				return min(1, frame['progress'] * multiplier)

	def reward(self, telemetry):
		return sum([self.frame_reward(frame) for frame in telemetry]) / float(len(telemetry))

	def save_all(self):
		self.save_model()
		print "Model saved at frame %d" % self.frame_count
		self.save_transitions()
		print "Transitions saved at frame %d" % self.frame_count

	def save_model(self):
		path = ""
		if _platform == "linux" or _platform == "linux2":
			path = "/data/q-net"
		elif _platform == "darwin":
			path = "./data/q-net"

		path = self.tfSaver.save(self.sess, path, self.frame_count)
		print "SAVED MODEL TO PATH: %s" % path

	def save_transitions(self):
		path = ""
		if _platform == "linux" or _platform == "linux2":
			path = "/data/transitions.pickle" 
		elif _platform == "darwin":
			path = "./data/transitions.pickle" 

		f = open(path, 'w+')
		to_dump = {'frame_count': self.frame_count, 'transitions': self.transitions}
		pickle.dump(to_dump, f)
		f.close()

	def save_initial(self):
		path = ""
		if _platform == "linux" or _platform == "linux2":
			path = "/data/transitions50k.pickle" 
		elif _platform == "darwin":
			path = "./data/transitions50k.pickle" 

		f = open(path, 'w+')
		to_dump = {'frame_count': self.frame_count, 'transitions': self.transitions}
		pickle.dump(to_dump, f)
		f.close()

	def epsilon(self):
		return ((hp.FINAL_EXPLORATION - hp.INITIAL_EXPLORATION) / hp.FINAL_EXPLORATION_FRAME) * self.frame_count + hp.INITIAL_EXPLORATION

	def step(self, image, reward, terminal, was_start, action, telemetry):
		t = time.time()
		print "Roundtrip from browser took %.2f ms" % ((t - self.timer[-1]) * 1000)

		self.frame_count += 1

		self.transitions.add_transition(image, terminal, action, reward, was_start, telemetry)

		if self.test_mode:
			img = self.transitions.get_recent()
			print "Getting recent transitions took %.2f ms" % ((time.time() - t) * 1000)
			t = time.time()
			best = Action(self.sess.run(best_action, feed_dict={s: img})[0])
			print "Forward pass took %.2f ms" % ((time.time() - t) * 1000)
			self.timer[-1] = time.time()
			return best

		if self.frame_count == hp.REPLAY_START_SIZE:
			self.save_initial()
		elif self.frame_count % hp.CHECKPOINT_FREQUENCY == 0 and self.frame_count < 250000:
			self.save_all()
		elif self.frame_count % hp.CHECKPOINT_FREQUENCY == 0:
			self.save_model()

		if self.frame_count < hp.REPLAY_START_SIZE:
			return Action.random_action()

		if self.frame_count % hp.UPDATE_FREQUENCY == 0:
			s_, t_, a_, r_, sp_ = self.transitions.get_minibatch(self.frame_count)
			minimize_loss_, loss_, avg_q_val_ = self.sess.run([minimize_loss, loss, avg_q_val], feed_dict={s: s_, sp: sp_, actions: a_, rewards: r_, terminals: t_})
			print "Loss: %.2f" % loss_
			print "Update took %.2f ms" % ((time.time() - t) * 1000)
			print "AverageQValue: %.2f" % avg_q_val_

		if self.frame_count % (hp.UPDATE_FREQUENCY * hp.TARGET_UPDATE_FREQUENCY) == 0:
			self.sess.run(update_target)

		if np.random.rand() < self.epsilon():
			return Action.random_action()
		t = time.time()
		img = self.transitions.get_recent()
		print "Getting recent transitions took %.2f ms" % ((time.time() - t) * 1000)
		t = time.time()
		best = Action(self.sess.run(best_action, feed_dict={s: img})[0])
		print "Forward pass took %.2f ms" % ((time.time() - t) * 1000)
		self.timer[-1] = time.time()
		return best
