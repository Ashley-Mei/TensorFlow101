from __future__ import print_function
import tensorflow as tf

# 学习教材：https://www.jianshu.com/p/840fa047e7a9

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)

# 在会话中查看节点数值
sess = tf.Session()
print(sess.run([node1,node2]))

# 操作也是一种节点
node3 = tf.add(node1, node2)
print(node3)
print(sess.run(node3))

# 占位符：参数化外部输入
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# 变量：添加可训练参数到图形中
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# 损失函数, e.g.线性回归的标准损失函数
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# 修改变量值
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# 优化器：梯度下降
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))


