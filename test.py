import numpy as np
from Neocognitron import Neocognitron
import tensorflow as tf
import numpy as np
import cv2
import tqdm

N = 100 # Number of examples to 'train' on
NUMER_TO_PREDICT = 5 # Between 0 and 9

# ------------------------------------------

# Import the mnist dataset from tensorflow
mnist = tf.keras.datasets.mnist
(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train / 255.0 # ???

# Convert from 28x28 to 19x19
x_train = np.array([cv2.resize(x, (19, 19)) for x in x_train])

if N % 10 != 0:
	print("Warning: N is not divisible by 10 (labels)")

# Make a dataset of N samples of each digit: 0-9 in the shape of (N, 19, 19)
x_train = np.array([x_train[y_train == i][:int(N/10)] for i in range(10)])
x_train = np.reshape(x_train, (N, 19,19))

def windower(pl_size, l_size):
	m = abs(pl_size-l_size)+1
	return (m, m)

layer_sizes = [(19, 12), (21, 8), (21, 38), (13, 19), (13, 35), (7, 23), (3, 11), (1, 10)]
windows = [(3, 3), windower(19, 21), (9, 9), windower(21, 13), (19, 19), windower(13, 7), (19, 19), windower(7, 3), (1, 1)]
n = Neocognitron(19, layer_sizes, windows)


# Forward training
history = []
count = 0
for i in tqdm.tqdm(range(N)):
	print('Forwarding number', count)
	history.append(n.estimate(x_train[i]))
	print([round(i[0][0],7) for i in history[-1]])
	count += 1
	if count == 9:
		count = 0

print('Forward training done.\n')
print('Predicting number', NUMER_TO_PREDICT)
b = n.estimate(x_train[NUMER_TO_PREDICT])
print([round(i[0][0],7) for i in b])
# Print the location of the list of the maximum value
print('The location of the maximum value is', np.argmax(b))