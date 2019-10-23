import pandas as pd
import os
import glob
import re

def to_words(raw):
    """
    Only keeps ascii characters in the tweet

    :param raw_tweet:
    :return:
    """
    letters_only = re.sub("[^a-zA-Z]", " ", raw)
    words = letters_only.lower().split()
    return " ".join(words)

def load_data(path):
    pos_dir = path + "pos/"
    pos_list = glob.glob(pos_dir+"*.txt")

    df = pd.DataFrame(columns=['text', 'label'])
    for file in pos_list:
        with open(file, 'r') as reader:
            content = reader.read()
        df = df.append({"text":to_words(content),"label":1}, ignore_index=True)

    neg_dir = path + "neg/"
    neg_list = glob.glob(neg_dir+"*.txt")
    for file in neg_list:
        with open(file, 'r') as reader:
            content = reader.read()
        df = df.append({"text":to_words(content),"label":0}, ignore_index=True)

    return df

# Load train data
print('Loading training...')
df_train = load_data("train/")
print('Writing training to csv...')
df_train.to_csv('imdb_train.csv', sep='\t')

print('Loading test...')
df_test = load_data("test/")
print('Writing test to csv...')
df_test.to_csv('imdb_test.csv', sep='\t')