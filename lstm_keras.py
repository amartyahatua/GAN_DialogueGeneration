import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import random
import numpy as np
import pandas as pd
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Input
from keras.models import Sequential
from keras.models import Model


class TARGET_LSTM_KERAS(object):

    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, sequence_length, start_token, params):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        #self.start_token = tf.constant(([start_token] * self.batch_size), ([start_token] * self.sequence_length),  dtype=tf.int32)
        self.start_token = np.zeros((self.batch_size, self.sequence_length))
        self.g_params = []
        self.temperature = 1.0
        self.params = params

        tf.set_random_seed(66)

        self.g_embeddings = tf.Variable(self.params[0])

        self.g_params.append(self.g_embeddings)

        print("*************************")
        print(self.start_token)

        print("*************************")
        print(self.g_embeddings)

        input_layer = Input(shape=(sequence_length,), name='text_input')
        embedding = Embedding(input_length=sequence_length, input_dim=num_emb, output_dim=emb_dim,
                               name='embedding_layer', trainable=False, weights=[self.params[0]])
        embedded_text = embedding(input_layer)

        embedding_model = Model(inputs=input_layer, outputs=embedded_text)

        # Random input sequence of length 200
        input_sequence = np.random.randint(0,num_emb,size=(1,sequence_length))

        # Extract the embeddings by calling the .predict() method
        sequence_embeddings = embedding_model.predict(input_sequence)



        #model.add(Embedding(num_emb, emb_dim, input_length=sequence_length, name='embedding_layer', trainable=False, weights=self.params[0]))
        #model.add(LSTM(sequence_length, kernel_initializer=self.params[1:14]))

        # #model.add(SpatialDropout1D(0.4))
        # model.add(LSTM(sequence_length, ))


        # input_context = Input(shape=(sequence_length,), dtype='int32', name='input_context')
        # input_current_token = Input(shape=(dictionary_size,), name='input_current_token')
        #
        # Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, input_length=maxlen_input, trainable=False, name = 'shared')
        # LSTM_encoder_discriminator = LSTM(sentence_embedding_size, init= 'lecun_uniform', name = 'encoder_discriminator')

