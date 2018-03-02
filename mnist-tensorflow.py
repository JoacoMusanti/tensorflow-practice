import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data", one_hot=True)

n_nodes_hidden_1 = 500
n_nodes_hidden_2 = 500
n_nodes_hidden_3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def nn(data):
    hidden_1 = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hidden_1])),
    'biases': tf.Variable(tf.random_normal([n_nodes_hidden_1]))}

    hidden_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_1, n_nodes_hidden_2])),
    'biases': tf.Variable(tf.random_normal([n_nodes_hidden_2]))}
        
    hidden_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_2, n_nodes_hidden_3])),
    'biases': tf.Variable(tf.random_normal([n_nodes_hidden_3]))}

    outputs = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_3, n_classes])),
    'biases': tf.Variable(tf.random_normal([n_classes]))}

    net_layer_1 = tf.add(tf.matmul(data, hidden_1['weights']), hidden_1['biases'])
    act_layer_1 = tf.nn.relu(net_layer_1)

    net_layer_2 = tf.add(tf.matmul(act_layer_1, hidden_2['weights']), hidden_2['biases'])
    act_layer_2 = tf.nn.relu(net_layer_2)

    net_layer_3 = tf.add(tf.matmul(act_layer_2, hidden_3['weights']), hidden_3['biases'])
    act_layer_3 = tf.nn.relu(net_layer_3)

    net_outputs = tf.add(tf.matmul(act_layer_3, outputs['weights']), outputs['biases'])

    return net_outputs

def train_nn(x):
    prediction = nn(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    # razon de aprendizaje = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    n_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for epoch in range(0, n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            
            print('Perdida para el paso {}: {}'.format(epoch, epoch_loss))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Precision: {}'.format(accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))

train_nn(x)

    