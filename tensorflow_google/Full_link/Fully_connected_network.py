import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

x = tf.placeholder(tf.float32,shape=(None,2),name='input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='output')

w1 = tf.Variable(tf.random_normal([2,3],seed=1),trainable=True)
w2 = tf.Variable(tf.random_normal([3,1],seed=1))
# x = tf.constant([[0.7,0.9]])

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
y = tf.sigmoid(y)



cross_entropy = -tf.reduce_mean(y*tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
learning_rate = tf.train.exponential_decay(0.001,)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2)<1] for (x1,x2) in X]


config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
with tf.Session(config=config) as sess:
    init_opt = tf.global_variables_initializer()
    sess.run(init_opt)

    steps = 5000
    for i in range(steps):
        start = i*(batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)

        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

        if i%100 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print('After '+str(i)+' training step (s) , cross entropy on all data is '+str(total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))