# Use Link Exsample
#http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/

import numpy
##from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.estimator import regression



# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
	seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
	print seq_in, '->', seq_out
	
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), seq_length))
# Sequence padding
X = pad_sequences(X, maxlen=1, value=0.)

Y = to_categorical(dataY,nb_classes=26)

# Network building
net = input_data(shape=[None,1])
net = embedding(net, input_dim=25, output_dim=5)
net = bidirectional_rnn(net,BasicLSTMCell(5), BasicLSTMCell(5))
#net = dropout(net, 0.5)
net = fully_connected(net, 26, activation='softmax')
net = regression(net, optimizer='adam', loss='categorical_crossentropy')
## Training
model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
model.fit(X, Y,n_epoch=2000, validation_set=0., show_metric=True, batch_size=64)

x = numpy.reshape([0],(1,1))
x = pad_sequences(x, maxlen=1, value=0.)
prediction = model.predict(x)
print prediction
index = numpy.argmax(prediction)
result = int_to_char[index]
seq_in = int_to_char[0]
print seq_in, "->", result

# demonstrate some model predictions
for pattern in dataX:
	x = numpy.reshape(pattern, (1,1))
	x = pad_sequences(x, maxlen=1, value=0.)
	prediction = model.predict(x)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print seq_in, "->", result
