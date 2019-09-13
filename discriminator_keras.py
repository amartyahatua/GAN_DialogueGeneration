from dataloader import Gen_Data_loader, Dis_dataloader
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential

seed=0

positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'

positive=pd.read_csv('save/real_data.csv', header=None)
negative=pd.read_csv('save/generator_sample.csv', header=None)

data=pd.concat([positive,negative], axis=0)


positiveCls=np.zeros((len(positive),2))
negativeCls=np.zeros((len(negative),2))

positiveCls[:,0]=1
negativeCls[:,1]=1

positive=positive.values.tolist()
negative=negative.values.tolist()

#positive=[positive,positiveCls]

data=data.values.tolist()
datacls=np.concatenate((positiveCls,negativeCls), axis=0)

print(len(data))
print(len(data[0]))

print(len(datacls))

print(datacls[0])

X_train, X_val, Y_train, Y_val=train_test_split(data, datacls, test_size=0.25, random_state=seed)

max_feature=5000
epoch=5
batch=32
emb_dim=32

print(X_train[0])
print(type(X_train))
#print(X_train.shape)
print(Y_train[0])

X_train=np.array(X_train)
Y_train=np.array(Y_train)
X_val=np.array(X_val)
Y_val=np.array(Y_val)




#print(X_train.shape)

#
model = Sequential()
model.add(Embedding(max_feature, emb_dim,input_length=20))
model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))

model.add(Dense(2, activation='sigmoid'))
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epoch, batch_size=batch, verbose=1)
