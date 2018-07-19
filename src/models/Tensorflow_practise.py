# GRAPH CODE

import tensorflow as tf
import numpy as np

# ==================
# Define the Graph
# ==================

# Define the Placeholders

X = tf.placeholder("float", [10, 10], name="X")  # our input is 10x10
Y1 = tf.placeholder("float", [10, 20], name="Y1")  # out output is 10x1
Y2 = tf.placeholder("float", [10, 20], name="Y2")  # out output is 10x1



# Define the wieights for the layers
initial_shared_layer_weights = np.random.rand(10, 20)
initial_Y1_layer_weights = np.random.rand(20, 20)
initial_Y2_layer_weights = np.random.rand(20, 20)



shared_layer_weights = tf.Variable(initial_shared_layer_weights, dtype="float32", name="Share_W")
Y1_layer_weights = tf.Variable(initial_Y1_layer_weights, dtype="float32", name="share_Y1")
Y2_layer_weights = tf.Variable(initial_Y2_layer_weights, dtype="float32", name="share_Y2")


#Contsruct the Layers with RELU Activations
shared_layer = tf.nn.relu(tf.matmul(X, shared_layer_weights))
Y1_layer = tf.nn.relu(tf.matmul(shared_layer, Y1_layer_weights))
Y2_layer = tf.nn.relu(tf.matmul(shared_layer, Y2_layer_weights))

# Calculate Loss
Y1_Loss = tf.nn.l2_loss(Y1 - Y1_layer)
Y2_Loss = tf.nn.l2_loss(Y2 - Y2_layer)
Joint_Loss = Y1_Loss + Y2_Loss

#Optimisers
Optimiser = tf.train.AdamOptimizer().minimize(Joint_Loss)
#Y1_op = tf.train.AdamOptimizer().minimize(Y1_Loss)
#Y2_op = tf.train.AdamOptimizer().minimize(Y2_Loss)

#Joint Trai

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(10):

        _, Joint_Loss = session.run(
            [Optimiser, Joint_Loss],
            {
                X: np.random.rand(10, 10) * 10,
                Y1: np.random.rand(10, 20) * 10,
                Y2: np.random.rand(10, 20) * 10
            })
        print(Joint_Loss)
