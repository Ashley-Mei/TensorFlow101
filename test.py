from __future__ import print_function
import tensorflow as tf

# 学习教材：https://www.jianshu.com/p/840fa047e7a9

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)

# 在会话中查看节点数值
sess = tf.Session()
print(sess.run([node1,node2]))

#操作也是一种节点
node3 = tf.add(node1, node2)
print(node3)
print(sess.run(node3))

#占位符：参数化外部输入
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b





