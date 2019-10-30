import datetime
import sys
from collections import Counter
from os import path

import pandas as pd
import numpy as np

from util import clean_document, load_configuration, plot_loss, plot_accuracy, save_model
from dl_tensor_board import TrainValTensorBoard

out_file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

if len(sys.argv) > 1:
    out_file_name = sys.argv[1]
if len(sys.argv) > 2:
    gpus_number = int(sys.argv[2])
else:
    gpus_number = 1

configuration = load_configuration('./configuration_glove.yaml')

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

# Load and transform to glove embeddings

# Load embedding matrix
embeddings_index = {}
glove_dir = configuration['embedding']['glove_path']
with open(glove_dir) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

# finally, vectorize the text samples into a 2D integer tensor
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_sequence_length = configuration['embedding']['max_sequence_length']
max_num_words = configuration['embedding']['max_num_words']

def text_to_seq(text):
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)

    word_index = tokenizer.word_index

    return pad_sequences(sequences, maxlen=max_sequence_length).tolist(), word_index

data_train['sequences'], word_index = text_to_seq(data_train['text'])
data_test['sequences'], _ = text_to_seq(data_test['text'])

# Prepare splits
from sklearn.model_selection import StratifiedShuffleSplit

train_idx, val_idx = next(StratifiedShuffleSplit(
                                n_splits=1,
                                test_size=configuration['data']['val_split']
                            ).split(
                                data_train['sequences'],
                                data_train['label']
                            )
                        )

val_X, val_y = data_train.iloc[val_idx]['sequences'],\
                data_train.iloc[val_idx]['label']
train_X, train_y = data_train.iloc[train_idx]['sequences'],\
                data_train.iloc[train_idx]['label']
test_X, test_y = data_test['sequences'],\
                data_test['label']


# prepare embedding matrix
embedding_dimension = configuration['embedding']['dimension']
num_words = min(max_num_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dimension))
for word, i in word_index.items():
    if i >= max_num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# Model Prepare
from keras.models import Sequential
from keras.layers import (Dense, Activation, Dropout,
                            Embedding,
                            LSTM, GRU, SimpleRNN)
from keras.optimizers import RMSprop, SGD, Adagrad, adam
from keras.initializers import Constant

model = Sequential()

# embedding

model.add(
    Embedding(
        num_words,
        embedding_dimension,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=max_sequence_length,
        trainable=False
    )
)

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
model.add(Dense(1, activation='sigmoid'))

# Train

Optimizer = eval(configuration['optimizer'])
optimizer = Optimizer(lr=configuration['lr'])
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

epochs = configuration['epochs']
batch_size = configuration['batch_size']

tensorboard_callback = TrainValTensorBoard(log_dir=path.join('./out/', out_file_name, 'tensorboard.log'))

history = model.fit(np.stack(train_X), train_y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(np.stack(val_X), val_y),
#            callbacks=[tensorboard_callback]
)


# Score
from sklearn.metrics import confusion_matrix, classification_report
import time


score, acc = model.evaluate(np.stack(test_X), test_y,
                            batch_size=batch_size
)
print()
print('Test ACC=', acc)

test_pred = model.predict_classes(np.stack(test_X))

print()
print('Confusion Matrix')
print('-'*20)
print(
    pd.DataFrame(
        confusion_matrix(test_y, test_pred),
        index=train_y.unique(), columns=train_y.unique()
    )
)
print()
print('Classification Report')
print('-'*40)
print(classification_report(test_y, test_pred))
print()
print("Ending:", time.ctime())

plot_accuracy(history, path.join('./result/', out_file_name))
plot_loss(history, path.join('./result/', out_file_name))
save_model(model, path.join('./out/', out_file_name))