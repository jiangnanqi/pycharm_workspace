import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1], name='v1'))
v2 = tf.Variable(tf.constant(2.0, shape=[1], name='v2'))
result = v1 + v2
# saver = tf.train.Saver()
# saver.export_meta_graph('./model/model.ckptmeda.json',as_text=True)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    with tf.gfile.GFile('./model/combined_model.pd', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
