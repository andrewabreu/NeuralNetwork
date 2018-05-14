import tensorflow as tf
import numpy as np


# Traduce una lista de etiquetas a un array de ceros y un uno.
# Este uno representa la neurona activada de la capa de salida.
# Por ejemplo, si la primera neurona es la que tiene que devolver
# una salida 1 para la etiqueta 1 entonces si tenemos tres neuronas
# en la capa de salida el resultado será [1.,0.,0.]

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


# Leemos los datos desde un fichero.
# ndarray de 150x5
data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading

# Desordenamos los datos.
# ndarray de 150x5
np.random.shuffle(data)  # we shuffle the data

# Datos de entrada a la red (características de cada muestra de flor de iris)
# ndarray de 150x4
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data

# Etiquetas
# input: ndarray de 150x1
# output: ndarray de 150x3
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code


# Imprimimos algunos ejemplos
print ("\nSome samples...")
for i in range(20):
    print (x_data[i], " -> ", y_data[i])
print

# Establecemos los marcadores de posición en el cual
# asignaremos los datos de cada lote durante el entrenamiento.
x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

# Establecemos las variables con las que se operará.
# Matriz de pesos y sesgo de la capa oculta.
W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)
# Matriz de pesos y sesgo de la capa de salida.
W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)
# Capa oculta
h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
# Capa de salida
y = tf.nn.softmax(tf.matmul(h, W2) + b2)
# Función de error
loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 20

for epoch in range(100):
    for jj in range(int(len(x_data) / batch_size)):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    print ("Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print (b, "-->", r)
    print ("----------------------------------------------------------------------------------")
