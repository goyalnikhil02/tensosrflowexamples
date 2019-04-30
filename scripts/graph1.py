import tensorflow as tf
import  numpy as np

np.random.seed(101)
tf.set_random_seed(101)

rand_a=np.random.uniform(0,100,(5,5))
#print(rand_a)

rand_b=np.random.uniform(0,100,(5,1))
#print(rand_b)

a=tf.placeholder(tf.float32)

b=tf.placeholder(tf.float32)

add_op = a+b
mul_op = a*b
add_result2=tf.add(1,2)
print(add_result2)

with tf.Session() as session:
    print(session.run(add_result2))
    #add_result=session.run(add_op,feed_dict={a:10,b:20})
    add_result = session.run(add_op, feed_dict={a: rand_a, b: rand_b})
    print(add_result)

    #mul_result=session.run(mul_op,feed_dict={a:10,b:20})
    mul_result = session.run(mul_op, feed_dict={a: rand_a, b: rand_b})
    print(mul_result)
