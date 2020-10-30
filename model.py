import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore")

data_dir = Path("corpus/")
train_data= pd.read_csv(data_dir / "train.csv")
test_data = pd.read_csv(data_dir / "test.csv")


columns_use = ['antagonise','antagonise:confidence','condescending','condescending:confidence',
              'dismissive','dismissive:confidence','generalisation','generalisation:confidence',
              'generalisation_unfair', 'generalisation_unfair:confidence','hostile',
               'hostile:confidence', 'sarcastic','sarcastic:confidence']
target_column = ['healthy']

X_train = train_data[columns_use]
y_train = train_data[target_column]
X_test = test_data[columns_use]
y_test = test_data[target_column]
len(X_train['antagonise'])

attributes = ['antagonise', 'condescending', 'dismissive', 'generalisation',
              'generalisation_unfair', 'hostile', 'sarcastic']
attributes_confidence = ['antagonise:confidence', 'condescending:confidence',
                         'dismissive:confidence', 'generalisation:confidence',
                         'generalisation_unfair:confidence', 'hostile:confidence',
                         'sarcastic:confidence']

attributes = ['antagonise', 'condescending', 'dismissive', 'generalisation',
              'generalisation_unfair', 'hostile', 'sarcastic']
confidence = ['antagonise:confidence', 'condescending:confidence',
              'dismissive:confidence', 'generalisation:confidence',
              'generalisation_unfair:confidence', 'hostile:confidence',
              'sarcastic:confidence']

attributes = ['antagonise', 'condescending', 'dismissive', 'generalisation',
              'generalisation_unfair', 'hostile', 'sarcastic']
confidence = ['antagonise:confidence', 'condescending:confidence',
              'dismissive:confidence', 'generalisation:confidence',
              'generalisation_unfair:confidence', 'hostile:confidence',
              'sarcastic:confidence']


def preprocess(X):
    for i in range(len(attributes)):
        X[attributes[i]] = X[attributes[i]] * X[confidence[i]] + (1 - X[attributes[i]]) * (1 - X[confidence[i]])


preprocess(X_train)
X_train = X_train.drop(confidence, axis=1)
preprocess(X_test)
X_test = X_test.drop(confidence, axis=1)

