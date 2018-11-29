import tensorflow as tf

# a = tf.Variable(tf.constant(1.0,shape=[1],name='v1'))
# b = tf.Variable(tf.constant(2.0,shape=[1],name='v2'))
# result = a+b
# init_op = tf.global_variables_initializer()

saver = tf.train.import_meta_graph('./model/model.ckpt.meta')

# saver = tf.train.import_meta_graph(tf.get_default_graph().get_tensor_by_name('add:0'))
with tf.Session() as sess:
    # sess.run(init_op)
    # saver.save(sess,'./model/model.ckpt')
    saver.restore(sess,'./model/model.ckpt')
    # print(result.eval())
    print(tf.get_default_graph().get_tensor_by_name('add:0').eval())