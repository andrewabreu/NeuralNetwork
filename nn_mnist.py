import gzip
import _pickle as cPickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mp

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding = 'latin1')
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

# plt.imshow(train_x[26].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample
# print (train_y[26])

# TODO: the neural net!!

x_data = train_x
y_data = one_hot(train_y, 10)

y_valid_data = one_hot(valid_y, 10)

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float" , [None, 10])

# W1 = tf.Variable(np.float32(np.random.rand(784, 20)) * 0.1)
# b1 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)
#
# W2 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)
# b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W1 = tf.Variable(np.float32(np.random.rand(784, 25)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(25)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(25, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

y = tf.nn.sigmoid(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

#train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
train = tf.train.GradientDescentOptimizer(0.02).minimize(loss)
#train = tf.train.GradientDescentOptimizer(0.03).minimize(loss)
#train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#train = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 10

error_list = []
accuracy_training_list = []

validation_error_list = []
accuracy_validation_list = []

epoch_list = []

threshold = 0.001
patience = 16
patience_count = 0

for epoch in range(150):
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})/batch_size
    error_list.append(error)

    validation_error = sess.run(loss, feed_dict={x: valid_x, y_: y_valid_data})/len(y_valid_data)
    validation_error_list.append(validation_error)

    epoch_list.append(epoch)

    print ("Epoch #:", epoch, "Error: ", error)
    print ("Epoch #:", epoch, "Error Valid: ", validation_error)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracy_training_result = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    accuracy_training_list.append(accuracy_training_result)
    print("Accuracy Training:" + str(accuracy_training_result))

    accuracy_validation_result = sess.run(accuracy, feed_dict={x: valid_x, y_: one_hot(valid_y, 10)})
    accuracy_validation_list.append(accuracy_validation_result)
    print("Accuracy Validation:" + str(accuracy_validation_result))

    result = sess.run(y, feed_dict={x: batch_xs})

    for b, r in zip(batch_ys, result):
        print (b, "-->", r)
    print ("----------------------------------------------------------------------------------")

    # Paramos el entrenamiento del modelo si se cumple esta condición
    pe = abs(validation_error_list[epoch] - validation_error_list[epoch-1])
    if epoch > 0 and pe < threshold:
        patience_count += 1
    else:
        patience_count = 0

    if patience_count > patience:
        print("Finalizamos el entrenamiento del modelo de forma temprana.")
        break

mp.title("Gráfica 1")
mp.plot(epoch_list,validation_error_list, label='Validation')
mp.plot(epoch_list,error_list, label='Training')
mp.xlabel('Número de épocas')
mp.ylabel('Error')
mp.legend()
mp.show()

mp.title("Gráfica 2")
mp.plot(epoch_list,accuracy_validation_list, label='Validation')
mp.plot(epoch_list,accuracy_training_list, label='Training')
mp.xlabel('Número de épocas')
mp.ylabel('Exactitud')
mp.legend()
mp.show()

print("---- Testing Model ----")
accuracy_testing_result = sess.run(accuracy, feed_dict={x: test_x, y_: one_hot(test_y,10)})
print ("Accuracy Testing = " + str(accuracy_testing_result))