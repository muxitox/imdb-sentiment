import pandas as pd

train_path = 'data/imdb_train.csv'
df = pd.read_csv(train_path, sep='\t')
print(df.head(3))