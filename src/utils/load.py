import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

this_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_path, '..', '..', 'data')
submissions_dir = os.path.join(this_path, '..', '..', 'submissions')


def get_train():
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    labels = train.target.values
    train = train.drop('id', axis=1)
    train = train.drop('target', axis=1)

    lbl_enc = preprocessing.LabelEncoder()
    labels = lbl_enc.fit_transform(labels)

    return train.values, labels


def get_train_test_split(test_size=0.33, random_state=None):
    X, y = get_train()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def encode_y(y):
    ret = np.zeros((y.shape[0], 9))
    for i in range(int(y.max())):
        ret[y == i, i] = 1
    return ret


def get_test():
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    test = test.drop('id', axis=1)
    return test.values


def gen_submission(fname, preds):
    sample = pd.read_csv(os.path.join(data_dir, 'sampleSubmission.csv'))
    submission = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])

    fpath = os.path.join(submissions_dir, fname + '.csv')
    submission.to_csv(fpath, index_label='id')
