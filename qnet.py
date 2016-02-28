import tensorflow as tf
import numpy as np
import hyperparameters as hp
from action import Action

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W, s):
  return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding="VALID")

with tf.name_scope("input"):
	input_images = tf.placeholder(tf.float32, shape=[
		None,
		hp.INPUT_SIZE,
		hp.INPUT_SIZE,
		hp.AGENT_HISTORY_LENGTH
	], name="input_images")


with tf.name_scope("conv_1"):
	W_conv1 = weight_variable([
		hp.CONV_1_SIZE,
		hp.CONV_1_SIZE,
		hp.AGENT_HISTORY_LENGTH,
		hp.CONV_1_DEPTH
	], name="W_conv1")
	b_conv1 = bias_variable([hp.CONV_1_DEPTH], name="b_conv1")
	h_conv1 = tf.nn.relu(conv2d(input_images, W_conv1, hp.CONV_1_STRIDE) + b_conv1)

with tf.name_scope("conv_2"):
	W_conv2 = weight_variable([
		hp.CONV_2_SIZE,
		hp.CONV_2_SIZE,
		hp.CONV_1_DEPTH,
		hp.CONV_2_DEPTH
	], name="W_conv2")
	b_conv2 = bias_variable([hp.CONV_2_DEPTH], name="b_conv2")
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, hp.CONV_2_STRIDE) + b_conv2)


with tf.name_scope("conv_3"):
	W_conv3 = weight_variable([
		hp.CONV_3_SIZE,
		hp.CONV_3_SIZE,
		hp.CONV_2_DEPTH,
		hp.CONV_3_DEPTH
	], name="W_conv3")
	b_conv3 = bias_variable([hp.CONV_3_DEPTH], name="b_conv3")
	h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, hp.CONV_3_STRIDE) + b_conv3)

conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])

with tf.name_scope("fc1"):
	W_fc1 = weight_variable([conv3_out_size, hp.FC_SIZE], name="W_fc1")
	b_fc1 = bias_variable([hp.FC_SIZE], name="b_fc1")
	resized = tf.reshape(h_conv3, [-1, conv3_out_size], name="resized")
	h_fc1 = tf.nn.relu(tf.matmul(resized, W_fc1) + b_fc1)


with tf.name_scope("out"):
	W_out = weight_variable([hp.FC_SIZE, Action.num_actions], name="W_out")
	b_out = bias_variable([Action.num_actions], name="b_out")
	actions = tf.matmul(h_fc1, W_out) + b_out

max_index = tf.argmax(actions, 1, name="max_index")

tf.histogram_summary("actions", actions)
