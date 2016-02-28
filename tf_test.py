import tensorflow as tf
import numpy as np
import scipy.misc
from qnet import actions, input_images, max_index

img = scipy.misc.imread("/Users/raphie/Downloads/tarese.jpg", mode="L")

with tf.Session() as sess:
	writer = tf.train.SummaryWriter("/tmp/bazbar_logs", sess.graph_def)
	sess.run(tf.initialize_all_variables())
	img = img[np.newaxis, :, :, np.newaxis]
	img = np.tile(img, 4)
	print img.shape
	[m, summary] = sess.run(
		[max_index, tf.merge_all_summaries()],
		feed_dict={input_images: img}
	)
	print m, summary
	writer.add_summary(summary, 0)
