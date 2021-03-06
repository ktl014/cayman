'''
dataset

Created on May 08 2018 11:55 
#@author: Kevin Le 
'''
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import lmdb
import caffe

#TODO Parse command for rawdata folder (filter for root)
ROOT = '/data6/lekevin/cayman'

def create_dataset(version=1):

    # Read in filename/label text files
    df = pd.DataFrame()
    rawdata_path = [os.path.join(ROOT,'rawdata','{}_SPC_Images_3-COLOR'.format(i)) for i in ['EC1', 'EC2', 'EC3']]
    for i in range(4):
        parse_param = {'sep':' ', 'names':['image', 'label', 'cool_example']}
        if i == 2:
            filename = os.path.join(ROOT, 'rawdata/yolksac.txt')
        elif i == 3:
            filename = '/data6/lekevin/cayman/rawdata/d3_predictions1.txt'
            parse_param = {'sep':',', 'names':['image', 'day', 'label']}
        else:
            filename = os.path.join(ROOT,'rawdata/classes_EC{}_1516combined.txt'.format(i+1))
        temp = pd.read_csv(filename, sep=parse_param['sep'], names=parse_param['names'], header=None)
        temp['day'] = ['EC{}'.format(i+1)] * temp.shape[0] if i != 3 else temp['day'].map ({'Thu Feb 16': 'EC3', 'Wed Feb 15': 'EC2', 'Fri Feb 17': 'EC3'})
        df = df.append(temp, ignore_index=True)

    # Map class labels to numeric labels
    with open (os.path.join (ROOT, 'rawdata/labels.txt')) as f:
        labels = {int (k): v for line in f for (k, v) in (line.strip ().split (None, 1),)}
    df['class'] = df['label'].map(labels)

    # Generate Unlabelled dataset
    img_list, days_list = [], []
    for i,dir in enumerate(rawdata_path):
        img_list += [img for img in os.listdir(dir)]
        days_list += ['EC{}'.format(i+1)] * len([img for img in os.listdir(dir)])
    unlbled_df = pd.DataFrame({'image': img_list, 'day': days_list, 'label': [0]*len(img_list)})
    test = unlbled_df[~unlbled_df['image'].isin (df['image'])].dropna ()  # Drop labeled data from df

    # Use only classes [1, 17, 7, 8, 14] for classification and other classes as class 6
    if version == 1:
        new_df = pd.DataFrame()
        minImgs = 200
        changed_labels = sorted([1, 7, 8, 14, 17])
        for i in range(1,20):
            temp = df[df['label'] == i]
            if i in changed_labels:
                temp.loc[temp['label'] == i, 'label'] = changed_labels.index(i) + 1
            else:
                # if temp size is less than 200, use all those images but constrain up to 200 images
                if temp.shape[0] > minImgs:
                    temp = temp.sample(n=200)

                temp.loc[temp['label'] == i, 'label'] = 6
            new_df = new_df.append (temp, ignore_index=True)

        # # Balance number of examples among class 6
        # image_counts = sorted(new_df['label'].value_counts(), reverse=True)
        # new_df = new_df.drop (new_df[new_df['label'] == 6].sample(n=image_counts[0] - image_counts[1]).index)
        #TODO Fix class labels
    elif version == 2:
        new_df = pd.DataFrame()
        changed_labels = sorted([1, 7, 8, 14, 17])
        for i in range(1,20):
            temp = df[df['label'] == i]
            if i in changed_labels:
                if i == 8:  # Jelly fish class
                    temp = temp.sample(n=1400)
                temp.loc[temp['label'] == i, 'label'] = changed_labels.index(i) + 1
                new_df = new_df.append (temp, ignore_index=True)
    elif version == 3:
        new_df = pd.DataFrame()
        minImgs = 50
        for i in range(1,20):
            temp = df[df['label'] == i]
            if i == 1:
                temp.loc[temp['label'] == i, 'label'] = 1
            else:
                if temp.shape[0] > minImgs:
                    temp = temp.sample(n=minImgs)
                temp.loc[temp['label'] == i, 'label'] = 0
            new_df = new_df.append (temp, ignore_index=True)
    elif version == 4:
        # Append predictions to it
        pred_df = pd.DataFrame ()
        for i in range (2):
            filename = os.path.join (ROOT, 'rawdata/d3_predictions{}.txt'.format (i))
            temp = pd.read_csv (filename, sep=',', names=['image', 'day', 'label'], header=None)
            pred_df = pred_df.append (temp, ignore_index=True)
        pred_df['day'] = pred_df['day'].map ({'Thu Feb 16': 'EC3', 'Wed Feb 15': 'EC2', 'Fri Feb 17': 'EC3'})
        df = df.append (pred_df)

        new_df = pd.DataFrame ()
        minImgs = 50
        for i in range (20):
            temp = df[df['label'] == i]
            if i == 0:
                temp.loc[temp['label'] == i, 'label'] = 0
            elif i == 1:
                temp.loc[temp['label'] == i, 'label'] = 1
            else:
                if temp.shape[0] > minImgs:
                    temp = temp.sample (n=minImgs)
                temp.loc[temp['label'] == i, 'label'] = 0
            new_df = new_df.append (temp, ignore_index=True)
    elif version == 5:
        # Uniform distribution of images among each class
        img_counts = df['label'].value_counts().to_dict()
        avg_img =int(df['label'].value_counts().mean())
        sampled_classes = {k:v for k,v in img_counts.items() if v > avg_img}
        for cls in sampled_classes:
            temp = df[df['label'] == cls].sample(n=sampled_classes[cls]-avg_img, random_state=123)
            df = df.drop(temp.index)
        new_df = df
    else:
        new_df = df

    # Partition dataset
    train, val, _, _ = train_test_split(new_df, new_df['label'], test_size=0.15, random_state=42)

    # Save dataset
    dest_path = os.path.join (ROOT, 'data/{}'.format (version))
    if not os.path.exists (dest_path):
        os.makedirs (dest_path)

    with open(dest_path + '/stats.txt', 'w') as f:
        dataset = {'train': train, 'val': val, 'test': test}
        for phase in ['train', 'val', 'test']:
            dataset[phase].to_csv(os.path.join(dest_path, 'data_{}.csv'.format(phase)))
            print('Dataset written to {}'.format(os.path.join(dest_path, 'data_{}.csv'.format(phase))))
            f.write('{} dataset\n'.format(phase))
            for key, val in dataset[phase]['label'].value_counts().to_dict().iteritems():
                f.write('{}: {}\n'.format(key, val))
    f.close()

class SPCDataset(object):
    def __init__(self, csv_filename, img_dir, phase):
        self.data = pd.read_csv(csv_filename)
        self.img_dir = img_dir
        self.data_dir = os.path.dirname(csv_filename)
        self.phase = phase
        self.size = self.data.shape[0]
        self.numclasses = len(self.data['label'].unique())
        self.lmdb_path = os.path.join(self.data_dir, '{}.LMDB'.format(self.phase))

        shuffle_images = (self.phase == 'train' or self.phase == 'val')
        if shuffle_images:
            self.data = self.data.iloc[np.random.permutation(self.size)]
            self.data = self.data.reset_index(drop=True)

        #TODO Give option to user to create lmdb after making dataset if he wants to
        # if not os.path.exists(self.lmdb_path):


    def __repr__(self):
        return 'Dataset [{}] {} classes, {} images\n{}'.\
            format(self.phase, self.numclasses, self.size, self.data['label'].value_counts())

    def get_fns(self):
        '''
        Return filenames and labels
        :return: fns: list, lbls: array
        '''

        # Append full path to images
        path_map = {i:os.path.join(self.img_dir, '{}_SPC_Images_3-COLOR'.format(i)) for i in self.data['day'].unique()}
        self.data['path'] = self.data['day'].map(path_map).astype(str) + '/' + self.data['image']

        self.fns = list(self.data['path'])
        self.lbls = np.array(self.data['label'])
        return self.fns, self.lbls

    def get_lmdbs(self):
        '''
        Get LMDBs (dataset version)
        :return: LMDB path
        '''
        # Catch if LMDBs exist or not
        try:
            return self.lmdb_path
        except:
            print('{} was not found or does not exist!'.format(self.lmdb_path))

    def load_lmdb(self):
        '''
        Load LMDB
        :param fn: filename of lmdb
        :return: images and labels
        '''
        print ("Loading " + str (self.lmdb_path))
        env = lmdb.open (self.lmdb_path, readonly=True)
        datum = caffe.proto.caffe_pb2.Datum ()
        with env.begin () as txn:
            cursor = txn.cursor ()
            data, labels = [], []
            for _, value in cursor:
                datum.ParseFromString (value)
                labels.append (datum.label)
                data.append (caffe.io.datum_to_array (datum).squeeze ())
        env.close ()
        print ("LMDB successfully loaded")
        return data, labels

    @staticmethod
    def map_labels(dataframe, label_file, mapped_column):
        with open(label_file, "r") as f:
            mapped_labels = {int(k): v for line in f for (k, v) in (line.strip ().split (None, 1),)}
        dataframe['class'] = dataframe[mapped_column].map(mapped_labels)
        return dataframe

if __name__ == '__main__':
    #TODO throw in catch function if dataset hasn't been created
    version = 5
    create_dataset(version=version)
    root = '/data6/lekevin/cayman'
    img_dir = '/data6/lekevin/cayman/rawdata'
    csv_filename = os.path.join(root, 'data', str(version), 'data_{}.csv')

    # Test initialization
    dataset = {phase: SPCDataset(csv_filename=csv_filename.format(phase), img_dir=img_dir, phase=phase) for phase in ['train', 'val', 'test']}
    for phase in dataset:
        print(dataset[phase])

    # Test file, lbl retrieval
    fns, lbls = dataset['train'].get_fns()