from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

n_inputs = 2
n_nodes_hidden_1 = 100
n_outputs = 2

# get the data set
data_x, data_y_prev = make_moons(n_samples=10000, shuffle=True, noise=0.3)

data_y_prime = []

# rearrange the data labels so that we can feed them to tensorflow
for i in range(0, len(data_y_prev)):
    if data_y_prev[i] == 1:
        data_y_prime.append(0)
    else:
        data_y_prime.append(1)

data_y = []

for i in range(len(data_y_prev)):
    data_y.append(data_y_prev[i])
    data_y.append(data_y_prime[i])

data_y = np.asarray(data_y).reshape(-1, 2)

# split the data into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

# tensorflow needs matrices
x_train = np.asmatrix(x_train)
y_train = np.asmatrix(y_train)
x_test = np.asmatrix(x_test)
y_test = np.asmatrix(y_test)

x = tf.placeholder('float', [None, 2])
y = tf.placeholder('float')


# generate a neural network with 1 hidden layer using a relu activation function
def generate_model(data):
    hidden_1 = {'weights': tf.Variable(tf.random_normal([n_inputs, n_nodes_hidden_1])),
                'biases': tf.Variable(tf.random_normal([n_nodes_hidden_1]))}

    outputs = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_1, n_outputs])),
               'biases': tf.Variable(tf.random_normal([n_outputs]))}

    net_1 = tf.add(tf.matmul(data, hidden_1['weights']), hidden_1['biases'])
    act_1 = tf.nn.relu(net_1)

    net_output = tf.add(tf.matmul(act_1, outputs['weights']), outputs['biases'])

    return net_output

# train the neural network using the adam optimizer with a cross entropy loss function
def train_model(x):
    prediction = generate_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        i = 0
        for data_item in x_train:
            data_label = y_train[i]
            _, c = sess.run([optimizer, cost], feed_dict={x: data_item, y: data_label})

            print('Perdida para el paso {}: {}'.format(i, c))
            i += 1

        # calculate and print the accuracy of the model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_test, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Precision: {}'.format(accuracy.eval({x: x_test, y: y_test})))

train_model(x)
