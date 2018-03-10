from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import tensorflow as tf

data_x, data_y = load_diabetes(return_X_y=True)

n_inputs = 10
n_outputs = 1
n_hidden_1 = 10
n_hidden_2 = 10

tf_x = tf.placeholder(tf.float32, [None, n_inputs])
tf_y = tf.placeholder(tf.float32)

def generate_model(data):
    hidden_1 = {'weights': tf.Variable(tf.random_normal([n_inputs, n_hidden_1])),
                'biases': tf.Variable(tf.random_normal([n_hidden_1]))}

    hidden_2 = {'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                'biases': tf.Variable(tf.random_normal([n_hidden_2]))}

    output = {'weights': tf.Variable(tf.random_normal([n_hidden_2, n_outputs])),
              'biases': tf.Variable(tf.random_normal([n_outputs]))}

    net_1 = tf.add(tf.matmul(data, hidden_1['weights']), hidden_1['biases'])
    act_1 = tf.nn.relu(net_1)

    net_2 = tf.add(tf.matmul(act_1, hidden_2['weights']), hidden_2['biases'])
    act_2 = tf.nn.relu(net_2)

    out = tf.add(tf.matmul(act_2, output['weights']), output['biases'])

    return out

def train_model(x):
    prediction = generate_model(x)
    cost = tf.reduce_mean(tf.nn.