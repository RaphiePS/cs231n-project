import tensorflow as tf
import numpy as np
import scipy.misc
import time
from double_qnet import r_actions, t_actions, s, sp, update_target, loss, actions, rewards, terminals, ys, minimize_loss, r_W_conv1, t_W_conv1

img = scipy.misc.imread("/Users/raphie/Downloads/tarese.jpg", mode="L")

with tf.Session() as sess:
	writer = tf.train.SummaryWriter("/tmp/bazbar_logs", sess.graph_def)
	sess.run(tf.initialize_all_variables())
	# img = img[np.newaxis, :, :, np.newaxis]
	# img = np.tile(img, 4)
	img = np.ones((32, 84, 84, 4))
	actions_ = np.ones(32)
	rewards_ = np.ones(32) * 0.5
	terminals_ = np.zeros(32)
	print map(np.linalg.norm, sess.run([r_W_conv1, t_W_conv1]))
	for i in range(100):
		t = time.time()
		sess.run([minimize_loss], feed_dict={s: img, sp: img, actions: actions_, rewards: rewards_, terminals: terminals_})
		print (time.time() - t) * 1000
	print map(np.linalg.norm, sess.run([r_W_conv1, t_W_conv1]))
	# [ar, at] = sess.run([r_actions, t_actions], feed_dict={s: img, sp: img})
	# print ar
	# print at
	# sess.run(update_target)
	# [ar, at] = sess.run([r_actions, t_actions], feed_dict={s: img, sp: img})
	# print ar
	# print at
	# print r_b_outa, t_b_outa, b_outa
