import tensorflow_datasets as tfds
import numpy as np


class Dataset:
    """
    Dataset class that allows for easier loading of a dataset
    """
    def __init__(self, name, load_func, label_list, train, valid, test):
        """
        Name: name of dataset
        Load: function for loading dataset
        Train: indexes of labels belonging to the training set
        Valid: indexes of labels belonging to the validation set
        Test: indexes of labels belonging to the test set
        Label List: All labels
        """
        self.name = name
        self.load = load_func
        self.train = train
        self.valid = valid
        self.test = test
        self.label_list = label_list
        
    def get_name(self):
        return self.name
    
    def load_data(self):
        """
        Will load the dataset at once
        """
        return self.load()
    
    def get_label_list(self):
        return self.label_list
    
    def get_train_labels(self):
        return self.train
    
    def get_valid_labels(self):
        return self.valid
    
    def get_test_labels(self):
        return self.test

    
    def get_label(self, index):
        return self.label_list[index]
    
    def make_mask(self, y, label_list):
        '''
        Return a mask representing the desired labels which are provided in label_list
        '''
        r = np.zeros(len(y), dtype=bool)
        for i in range(len(y)):
            if y[i] in label_list:
                r[i] = True
        return r    
    
    def split(self, x = None, y = None):
        """
        Splits the dataset into train, valid, and test, depending on the masks provided at construction
        """
        assert (x != None and y != None) or (x == None and y == None)
        if x is None:
            x, y = self.load_data()
            
        train_mask = self.make_mask(y, self.train)
        valid_mask = self.make_mask(y, self.valid)
        test_mask = self.make_mask(y, self.test)
        return (x[train_mask], y[train_mask]), (x[valid_mask], y[valid_mask]), (x[test_mask], y[test_mask])

################################################################
# Building emnist/balanced
###############################################################
def load_emnist_balanced():
    """ 
    Loading EMNIST Balanced dataset
    """
    data = tfds.load('emnist/balanced')
    data_x = [0] * (len(data['train']) + len(data['test']))
    data_y = [0] * len(data_x)
    for example, i in zip(data['train'], range(len(data['train']))):
        data_x[i] = example['image']
        data_y[i] = example['label']
    for example, i in zip(data['test'], range(len(data['train']), len(data_x))):
        data_x[i] = example['image']
        data_y[i] = example['label']
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_x = np.squeeze(data_x, axis=3)
    data_y = np.expand_dims(data_y, axis=1)
    data_x = data_x.astype("float64") / 255

    return data_x, data_y

def make_emnist_balanced(labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C/c', 'D', 'E', 'F', 'G', 'H', 'I/i', 'J/j', 'K/k', 'L/l', 'M/m', 'N', 'O/o', 'P/p', 'Q', 'R', 'S/s', 'T', 'U/u', 'V/v', 'W/w', 'X/x', 'Y/y', 'Z/z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'], train = [7, 33, 44, 45, 29, 3, 22, 8, 24, 11, 18, 31, 25, 15, 14, 4, 1, 36, 17, 46, 30, 27, 43, 34, 42, 23, 39, 9, 38, 13, 26, 5, 20], valid = [6, 28, 12, 40, 32, 10, 19, 41, 37], test = [0, 2, 35, 16, 21]):
    
    return Dataset('emnist/balanced', load_emnist_balanced, labels, train=train, valid=valid, test=valid)


################################################################
# Building dataset loaders
################################################################

dataset_dict = {'emnist/balanced' : make_emnist_balanced}

def get_datasets():
    return dataset_dict.keys()

def load_dataset(db_name):
    """ 
    USE THIS ONE!
    Pass in emnist/balanced to load it!
    """
    return dataset_dict[db_name]()
    