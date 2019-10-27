import os
import pickle
import re
from os import path

import yaml
from keras.models import model_from_json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_configuration(path='./configuration.yaml'):
    with open(path, 'r') as f:
        return yaml.load(f)

def clean_document(document):
    words = re.sub(r'[^a-zA-Z]', ' ', document).lower()
    return words

def plot_accuracy(history, base_folder='./'):
    make_base_folder_if_needed(base_folder)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig(path.join(base_folder, 'accuracy.pdf'))
    plt.close()


def plot_loss(history, base_folder='./'):
    make_base_folder_if_needed(base_folder)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig(path.join(base_folder, 'loss.pdf'))

def save_model(model, base_folder='./'):
    make_base_folder_if_needed(base_folder)
    model_json = model.to_json()
    with open(path.join(base_folder, 'model.json'), 'w') as json_file:
            json_file.write(model_json)
    weights_file = path.join(base_folder, 'weights.hdf5')
    model.save_weights(weights_file, overwrite=True)

def save_history(history, base_folder='./'):
    make_base_folder_if_needed(base_folder)
    with open(path.join(base_folder, 'history.pickle'), 'wb') as history_file:
        pickle.dump({
            'train_acc': history.history['acc'],
            'train_loss': history.history['loss'],
            'val_acc': history.history['val_acc'],
            'val_loss': history.history['val_loss']
        }, history_file)

def make_base_folder_if_needed(base_folder):
    try:
        os.makedirs(base_folder)
    except:
        pass
