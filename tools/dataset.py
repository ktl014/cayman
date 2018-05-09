'''
dataset

Created on May 08 2018 11:55 
#@author: Kevin Le 
'''
import pandas as pd
import os
from sklearn.model_selection import train_test_split


ROOT = '/data6/lekevin/cayman'

def create_dataset(version=1):

    # Read in filename/label text files
    df = pd.DataFrame()
    for i in range(1,4):
        if i == 3:
            filename = os.path.join(ROOT, 'rawdata/yolksac.txt')
        else:
            filename = os.path.join(ROOT,'rawdata/classes_EC{}_1516combined.txt'.format(i))
        temp = pd.read_csv(filename, sep=' ', names=['image', 'label', 'cool_example'], header=None)
        temp['day'] = ['EC{}'.format(i)] * temp.shape[0]
        df = df.append(temp, ignore_index=True)

    # Map class labels to numeric labels
    with open (os.path.join (ROOT, 'rawdata/labels.txt')) as f:
        labels = {int (k): v for line in f for (k, v) in (line.strip ().split (None, 1),)}
    df['class'] = df['label'].map(labels)

    # Use only classes [1, 17, 7, 8, 14] for classification and other classes as class 6
    if version == 1:
        new_df = pd.DataFrame()
        changed_labels = [1, 7, 8, 14, 17]
        for i in range(1,20):
            temp = df[df['label'] == i]
            if i in changed_labels:
                temp.loc[temp['label'] == i, 'label'] = changed_labels.index(i) + 1
            else:
                temp.loc[temp['label'] == i, 'label'] = 6
            new_df = new_df.append (temp, ignore_index=True)
    else:
        new_df = df

    # Partition dataset
    train, val, _, _ = train_test_split(new_df, new_df['label'], test_size=0.15, random_state=42)

    # Save dataset
    dest_path = os.path.join (ROOT, 'data/{}'.format (version))
    if not os.path.exists (dest_path):
        os.makedirs (dest_path)

    with open(dest_path + '/stats.txt', 'w') as f:
        dataset = {'train': train, 'val': val}
        for phase in ['train', 'val']:
            dataset[phase].to_csv(os.path.join(dest_path, 'data_{}.csv'.format(phase)))
            f.write('{} dataset\n'.format(phase))
            for key, val in dataset[phase]['label'].value_counts().to_dict().iteritems():
                f.write('{}: {}\n'.format(key, val))
    f.close()


if __name__ == '__main__':
    create_dataset()