import tensorflow as tf

a = tf.constant(10)

hello = tf.constant("Hello")

fill_mat = tf.fill((4, 4), 10)
myZero = tf.zeros((4, 4))
myrandom2 = tf.random_uniform((4, 4), minval=0, maxval=5)

my_ops = [a, hello, fill_mat, myZero,  myrandom2]



with tf.Session() as session :
    for ops in my_ops:
        print(session.run(ops))
        print("\n")

