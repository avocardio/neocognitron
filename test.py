import numpy as np
from Neocognitron import Neocognitron
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

# Import the mnist dataset from tensorflow
mnist = tf.keras.datasets.mnist
(x_train, y_train), (_, _) = mnist.load_data()
# x_train = x_train / 255.0 # ???

# Convert from 28x28 to 19x19
x_train = np.array([cv2.resize(x, (19, 19)) for x in x_train])

# Make a dataset of 10 samples of each digit: 0-9 in the shape of (10, 19, 19)
x_train = np.array([x_train[y_train == i][:1] for i in range(10)])
x_train = np.reshape(x_train, (10, 19, 19))
y_train = np.array([i for i in range(10)])

def windower(pl_size, l_size):
	m = abs(pl_size-l_size)+1
	return (m, m)

layer_sizes = [(19, 12), (21, 8), (21, 38), (13, 19), (13, 35), (7, 23), (3, 11), (1, 10)]
windows = [(3, 3), windower(19, 21), (9, 9), windower(21, 13), (19, 19), windower(13, 7), (19, 19), windower(7, 3), (1, 1)]
n = Neocognitron(19, layer_sizes, windows)



# Forward training
a = []
for i in range(10):
	print('Forwarding number', i)
	a.append(n.estimate(x_train[i]))
	print(a[-1])

print('Forward training done.\n')
print('Predicting number 5')
b = n.estimate(x_train[5])
print(b)
# Print the location of the list of the maximum value
print('The location of the maximum value is', np.argmax(b))