import tensorflow as tf

n1=tf.constant(1)

n2=tf.constant(2)

n3=n1+n2

with tf.Session() as session:
    result=session.run(n3)


print(result)
print(n3)

graph_one=tf.Graph()
print(graph_one)

graph_two=tf.get_default_graph()
print(graph_two)

graph_two_default=tf.get_default_graph()
print(graph_two_default)

graph_three=tf.Graph()
print(graph_three)