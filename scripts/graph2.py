import tensorflow as tf
import numpy as np


n_features=10  #row or say feauters or attribute

n_dense_neuron=3  #layer


#weight
x=tf.placeholder(tf.float32,(None,n_features))
#print(x)

#matrix data
W = tf.Variable(tf.random_normal([n_features,n_dense_neuron]))
#print(W)

c = tf.Variable(tf.ones([n_dense_neuron]))
#print(c)

xW = tf.matmul(x,W)

#print(xW)

z = tf.add(xW,c)

a=tf.sigmoid(z)

init=tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    layer_out=session.run(a,feed_dict={x:np.random.random([1,n_features])})
    print(layer_out)
