import tensorflow as tf
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import random
import numpy as np
import pandas as pd
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential




model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be
# no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
print(output_array)
assert output_array.shape == (32, 10, 64)

print("*************************888")
x = Embedding(1000, 64, input_array)
print(x)
