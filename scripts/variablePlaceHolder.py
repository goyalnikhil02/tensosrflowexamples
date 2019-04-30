import  tensorflow as tf


my_tensor=tf.random_uniform((4,4),0,1)
#this is tensor
print(my_tensor)

#this is a variable
my_variable=tf.Variable(initial_value=my_tensor)

my_variable2=tf.Variable(initial_value="Nikhil")


print(my_variable)
print(my_variable2)

#place holder in tensor
ph=tf.placeholder("float",None)
y=ph*2

print(ph)


ph2=tf.placeholder(tf.float32,)

init=tf.global_variables_initializer()

with tf.Session() as session :
    session.run(init)
    print(session.run(my_variable))
    print("\n")
    print(session.run(my_variable2))
    result = session.run(y, feed_dict={ph: [1, 2, 3]})
    print("\n")
    print(result)
