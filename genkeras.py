from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Dropout, merge
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.preprocessing import sequence

import keras.backend as K
import numpy as np
from numpy import int32

np.random.seed(1234)  # for reproducibility
import cPickle
import theano
import os.path
import sys
#import nltk
# import re
# import time




word_embedding_size = 100
sentence_embedding_size = 300
dictionary_size = 7000
maxlen_input = 50


maxlen_input=20
vocab_size=5000
maxlen_output=20






input_context = Input(shape=(maxlen_input,), dtype='int32', name='the_context_text')
Shared_Embedding = Embedding(output_dim=maxlen_output, input_dim=vocab_size, input_length=maxlen_input, name='Shared')
LSTM_encoder = LSTM(sentence_embedding_size, init= 'lecun_uniform', name='Encode_context')


word_embedding_context = Shared_Embedding(input_context)
context_embedding = LSTM_encoder(word_embedding_context)
out = Dense(dictionary_size, activation="relu", name='relu_activation')(context_embedding)
out = Dense(dictionary_size, activation="softmax", name='likelihood_of_the_current_token_using_softmax_activation')(out)

model = Model(input=[input_context], output = [out])
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00005, clipvalue=5))


#input=K.random_normal([64,20], mean=0.0, stddev=1.0, dtype=None, seed=None)
input = np.random.randint(low=1, high=5000, size=[64,20], dtype=int32)

#np.random.randint
print(input)


print("***************************************")
print(model.predict(input))

result=model.predict(input)
minv=np.min(result)
maxv=np.max(result)

print(maxv)
print(minv)

result = (result-minv)/(maxv-minv)

#result = int(result*5000)
result.astype(int)
print("***************************************")
print(result)




word_embedding_size = 100
sentence_embedding_size = 300
dictionary_size = 7000
maxlen_input = 50


maxlen_input=20
vocab_size=5000
maxlen_output=20






input_context = Input(shape=(maxlen_input,), dtype='int32', name='the_context_text')
Shared_Embedding = Embedding(output_dim=maxlen_output, input_dim=vocab_size, input_length=maxlen_input, name='Shared')
LSTM_encoder = LSTM(sentence_embedding_size, init= 'lecun_uniform', name='Encode_context')


word_embedding_context = Shared_Embedding(input_context)
context_embedding = LSTM_encoder(word_embedding_context)
out = Dense(dictionary_size, activation="relu", name='relu_activation')(context_embedding)
out = Dense(dictionary_size, activation="softmax", name='likelihood_of_the_current_token_using_softmax_activation')(out)

model = Model(input=[input_context], output = [out])
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00005, clipvalue=5))


#input=K.random_normal([64,20], mean=0.0, stddev=1.0, dtype=None, seed=None)
input = np.random.randint(low=1, high=5000, size=[64,20], dtype=int32)

#np.random.randint
print(input)


print("***************************************")
print(model.predict(input))

result=model.predict(input)
minv=np.min(result)
maxv=np.max(result)

print(maxv)
print(minv)

result = ((result-minv)*5000/(maxv-minv))
#result = int(result)
#result = int(result*5000)
#result.astype(int)

for i in range(len(result)):
    for j in range(len(result[0])):
        result[i,j] = int(result[i,j])
print("***************************************")
print(result)
