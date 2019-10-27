from collections import Counter
from os import path

import pandas as pd
import numpy as np

from util import clean_document, load_configuration

configuration = load_configuration()

train_path = path.join(
    configuration['data']['folder'],
    configuration['data']['file_train']
)

test_path = path.join(
    configuration['data']['folder'],
    configuration['data']['file_test']
)

data_train = pd.read_csv(train_path)
data_test = pd.read_csv(test_path)

# Clean Data

data_test['text'] = data_test['text'].apply(clean_document)
data_train['text'] = data_train['text'].apply(clean_document)

# Convert words to integers

counts = Counter(' '.join(data_train['text']).split())

numwords = configuration['embedding']['n_word']
vocab = sorted(counts, key=counts.get, reverse=True)[:numwords]
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

data_train['vectorized_text'] = data_train['text'].apply(
    lambda text: [vocab_to_int[word] for word in text.split() if word in vocab_to_int]
)

data_test['vectorized_text'] = data_test['text'].apply(
    lambda text: [vocab_to_int[word] for word in text.split() if word in vocab_to_int]
)

# Drop Zero Length Documents

data_train.drop(
    data_train[data_train['vectorized_text'].apply(lambda l: len(l) < 0)].index
)

data_test.drop(
    data_test[data_test['vectorized_text'].apply(lambda l: len(l) < 0)].index
)

# Pad documents

all_documents = data_train['vectorized_text'] + data_test['vectorized_text']
max_len = max(all_documents.apply(len))

data_train['padded_vectorized_text'] = data_train['vectorized_text'].apply(
    lambda l: [0 if i >= len(l) else l[i] for i in range(max_len)]
)
data_test['padded_vectorized_text'] = data_test['vectorized_text'].apply(
    lambda l: [0 if i >= len(l) else l[i] for i in range(max_len)]
)

# Prepare splits

val_elems = int(len(data_train) * configuration['data']['val_split'])

val_X, val_y = data_train.iloc[: val_elems]['padded_vectorized_text'],\
                data_train.iloc[: val_elems]['label']
train_X, train_y = data_train.iloc[val_elems :]['padded_vectorized_text'],\
                data_train.iloc[val_elems :]['label']
test_X, test_y = data_test['padded_vectorized_text'],\
                data_test['label']

# Model Prepare
from keras.models import Sequential
from keras.layers import (Dense, Activation, Dropout,
                            Embedding,
                            LSTM, GRU, SimpleRNN)
from keras.optimizers import RMSprop, SGD, Adagrad, adam

model = Sequential()

# embedding
embedding_size = configuration['embedding']['size']
model.add(Embedding(numwords + 1, embedding_size, input_length=max_len))

# rnn

RNN = eval(configuration['rnn']['type'])
n_layers = configuration['rnn']['n_layers']
dropout = configuration['rnn']['dropout']
for i in range(n_layers - 1):
    neurons = configuration['rnn']['neurons'][i]
    model.add(RNN(neurons, recurrent_dropout=dropout, return_sequences=True))
model.add(RNN(configuration['rnn']['neurons'][-1], recurrent_dropout=dropout))

fully_connected_dropout = configuration['dropout']
if fully_connected_dropout:
    model.add(Dropout(fully_connected_dropout))
model.add(Dense(len(train_y.unique())))

# Train

from keras.utils.np_utils import to_categorical
# import ipdb; ipdb.set_trace()

Optimizer = eval(configuration['optimizer'])
optimizer = Optimizer(lr=configuration['lr'])
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

epochs = configuration['epochs']
batch_size = configuration['batch_size']


model.fit(np.stack(train_X), to_categorical(train_y),
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(np.stack(val_X.values), to_categorical(val_y))
)

# Score
from sklearn.metrics import confusion_matrix, classification_report
import time


score, acc = model.evaluate(np.stack(test_X), to_categorical(test_y),
                            batch_size=batch_size
)
print()
print('Test ACC=', acc)

test_pred = model.predict_classes(np.stack(test_X))

print()
print('Confusion Matrix')
print('-'*20)
print(confusion_matrix(test_y, test_pred))
print()
print('Classification Report')
print('-'*40)
print(classification_report(test_y, test_pred))
print()
print("Ending:", time.ctime())