from tensorflow import keras
import numpy as np

class TripletSequence(keras.utils.Sequence):
    '''
    The skeleton code of the sequence is based on code from: https://stackoverflow.com/questions/70230687/how-keras-utils-sequence-works
    '''
    def __init__(self, x_in, y_in, samples_per_label=1, batch_size=5, can_shuffle=True, is_generator=True):
        '''
        samples_per_label: the total number of triplet is samples per label (positive) * (the number of labels-1) * samples per label
                           if the number of samples per label is 2 and the number of labels is 33 is we are assembling 2 * 32 * 2 triplets.
        '''
        # Initialization
        self.batch_size  = batch_size
        self.can_shuffle = can_shuffle
        self.x = x_in
        self.y = y_in
        self.samples_per_label = samples_per_label
        self.label_list = np.unique(self.y)
        self.is_generator = is_generator
        
        self.datalen = len(y_in)
        self.indexes = np.arange(self.datalen)
        if self.can_shuffle:
            np.random.shuffle(self.indexes)
            
        self.index_table = {}
        self.make_pstv_and_ngtv_indexes()
        self.make_triplet_matrix()
    
    def make_pstv_and_ngtv_indexes(self):
        '''
        Make two lists: (1) a list of label indexes and () a list of all other labels
        '''
        for label in self.label_list:
            assert label in self.y, f'Label {label} is not a valid class.'
            pstv_array = np.where(self.y == label)[0]
            ngtv_array = np.where(self.y != label)[0]

            self.index_table[label] = (pstv_array, ngtv_array)
    
    def make_triplet_indexes(self, label):
        '''
        Return three index arrays per a label: (1) the anchor indexes, (2) the positive indexes, snf (3) the negative indexes.
        '''
        assert label in self.label_list, f'Label {label} is not a valid class.'
        
        pstv_array, ngtv_array = self.index_table[label]
        
        a_array = np.copy(pstv_array)
        np.random.shuffle(a_array)
        
        p_array = np.copy(pstv_array)
        np.random.shuffle(p_array)
        
        n_array = np.copy(ngtv_array)
        np.random.shuffle(n_array)
        n_array = n_array[0:len(pstv_array)]
        
        assert len(a_array) == len(p_array), f'The anchor and the positive arrays must have the same length.'
        assert len(p_array) == len(n_array), f'The negative and the positive arrays must have the same length.'
        
        assert self.y[a_array[0]] == label, 'The anchor must have the desired label.'
        assert self.y[p_array[0]] == label, 'The positive must have the desired label.'
        assert self.y[n_array[0]] != label, 'The negative must not have the desired label.'
        
        return a_array, p_array, n_array
    
    def make_triplet_matrix(self):
        '''
        Make a matrix where its first column is the anchor indexes, the second column is the positive  indexes, and
        the third column is the negative indexes. 
        This matrix is shuffled.
        '''
        self.matrix = np.ones((self.datalen, 3), dtype=int) * -1
        
        next_start = 0
        for a_label in self.label_list:
            a_array, p_array, n_array = self.make_triplet_indexes(a_label)
            next_end = next_start + len(a_array)
            self.matrix[next_start:next_end, 0] = a_array
            self.matrix[next_start:next_end, 1] = p_array
            self.matrix[next_start:next_end, 2] = n_array
            next_start = next_end
            
        np.random.shuffle(self.matrix)
        
        assert len(np.where(self.matrix == -1)[0]) == 0, 'Something wrong with the triplet matrix.'
            
            
    def __getitem__(self, index):    
        # Determine batch start and end
        batch_start = index*self.batch_size
        batch_end   = (index+1)*self.batch_size
        if batch_end > self.datalen:
            batch_end = self.datalen
        batch_size = batch_end - batch_start
            
        # Allocate tensors to hold x and y for the batch
        _, num_row, num_col = self.x.shape
        x_tensor = np.zeros((batch_size, num_row, num_col, 3))
        
        # Collect images into the tensors    
        x_tensor[:, :, :, 0] = self.x[self.matrix[batch_start:batch_end, 0], :, :]
        x_tensor[:, :, :, 1] = self.x[self.matrix[batch_start:batch_end, 1], :, :]
        x_tensor[:, :, :, 2] = self.x[self.matrix[batch_start:batch_end, 2], :, :]
            
        if self.is_generator: 
            return x_tensor, x_tensor
        else:
            # Collect labels into the tensors
            y_tensor = np.zeros((batch_size, 2))
            y_tensor[:, 0] = np.squeeze(self.y[self.matrix[batch_start:batch_end, 0]]) # The anchor and the positive labels
            y_tensor[:, 1] = np.squeeze(self.y[self.matrix[batch_start:batch_end, 2]]) # The negative labels
            return x_tensor, y_tensor
            
    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        # Make new triplet indexes at the end of each epoch
        if self.can_shuffle:
            self.make_triplet_matrix()
            
            
class PairSequence(keras.utils.Sequence):
    '''
    The skeleton code of the sequence is based on code from: https://stackoverflow.com/questions/70230687/how-keras-utils-sequence-works
    '''
    def __init__(self, x_in, y_in, samples_per_label=1, batch_size=5, can_shuffle=True, output="x"):
        '''
        samples_per_label: the total number of triplet is samples per label (positive) * (the number of labels-1) * samples per label
                           if the number of samples per label is 2 and the number of labels is 33 is we are assembling 2 * 32 * 2 triplets.
        '''
        # Initialization
        self.batch_size  = batch_size
        self.can_shuffle = can_shuffle
        self.x = x_in
        self.y = y_in
        self.samples_per_label = samples_per_label
        self.label_list = np.unique(self.y)
        assert output in ['x', 'y', 'xy'], f'The output must be x, y, or xy: recevied {output}.'
        self.output = output
    
        self.datalen = len(y_in)
        self.indexes = np.arange(self.datalen)
        if self.can_shuffle:
            np.random.shuffle(self.indexes)
            
        self.index_table = {}
        
        self.make_pstv_and_ngtv_indexes()
        self.make_pair_matrix()
    
    def make_pstv_and_ngtv_indexes(self):
        '''
        Make two lists: (1) a list of label indexes and () a list of all other labels
        '''
        for label in self.label_list:
            assert label in self.y, f'Label {label} is not a valid class.'
            pstv_array = np.where(self.y == label)[0]
            ngtv_array = np.where(self.y != label)[0]

            self.index_table[label] = (pstv_array, ngtv_array)
    
    def make_triplet_indexes(self, label):
        '''
        Return three index arrays per a label: (1) the anchor indexes, (2) the positive indexes, snf (3) the negative indexes.
        '''
        assert label in self.label_list, f'Label {label} is not a valid class.'
        
        pstv_array, ngtv_array = self.index_table[label]
        
        a_array = np.copy(pstv_array)
        np.random.shuffle(a_array)
        
        p_array = np.copy(pstv_array)
        np.random.shuffle(p_array)
        
        n_array = np.copy(ngtv_array)
        np.random.shuffle(n_array)
        n_array = n_array[0:len(pstv_array)]
        
        assert len(a_array) == len(p_array), f'The anchor and the positive arrays must have the same length.'
        assert len(p_array) == len(n_array), f'The negative and the positive arrays must have the same length.'
        
        assert self.y[a_array[0]] == label, 'The anchor must have the desired label.'
        assert self.y[p_array[0]] == label, 'The positive must have the desired label.'
        assert self.y[n_array[0]] != label, 'The negative must not have the desired label.'
        
        return a_array, p_array, n_array
    
    def make_pair_matrix(self):
        '''
        Make a matrix where its first column is the anchor indexes, the second column is the positive  indexes,
        or the negative indexes.
        Make the corresponding label vector: 1 means a similar pair and 0 means a dissimilar pair. 
        This matrix is shuffled.
        '''
        self.matrix      = np.ones((2 * self.datalen, 2), dtype=int) * -1
        self.pair_labels = np.ones((2 * self.datalen, 3), dtype=int) * -1
        
        next_start = 0
        for a_label in self.label_list:
            a_array, p_array, n_array = self.make_triplet_indexes(a_label)
    
            next_end = next_start + len(a_array)
            
            self.matrix[next_start:next_end, 0] = a_array
            self.matrix[next_start:next_end, 1] = p_array
            self.pair_labels[next_start:next_end, 0] = np.squeeze(self.y[a_array])
            self.pair_labels[next_start:next_end, 1] = np.squeeze(self.y[p_array])
            self.pair_labels[next_start:next_end, 2] = 1
            assert np.array_equal(self.pair_labels[next_start:next_end, 0], self.pair_labels[next_start:next_end, 1])
            
            next_start = next_end
            next_end = next_start + len(a_array)
            
            self.matrix[next_start:next_end, 0] = a_array
            self.matrix[next_start:next_end, 1] = n_array
            self.pair_labels[next_start:next_end, 0] = np.squeeze(self.y[a_array])
            self.pair_labels[next_start:next_end, 1] = np.squeeze(self.y[n_array])
            self.pair_labels[next_start:next_end, 2] = 0
            assert not np.array_equal(self.pair_labels[next_start:next_end, 0], self.pair_labels[next_start:next_end, 1])
            
            next_start = next_end
            
        rand_perm = np.random.permutation(len(self.matrix))  
        self.matrix = self.matrix[rand_perm, ...]
        self.pair_labels = self.pair_labels[rand_perm]
                
        assert len(np.where(self.matrix == -1)[0]) == 0, 'Something wrong with the pair matrix.'
        assert len(np.where(self.pair_labels == -1)[0]) == 0, 'Something wrong with the pair labels.'   
            
    def __getitem__(self, index):    
        # Determine batch start and end
        batch_start = index*self.batch_size
        batch_end   = (index+1)*self.batch_size
        if batch_end > len(self.matrix):
            batch_end = len(self.matrix)
        batch_size = batch_end - batch_start
            
        # Allocate tensors to hold x and y for the batch
        _, num_row, num_col = self.x.shape
        x_tensor = np.zeros((batch_size, num_row, num_col, 2))
        
        # Collect images into the tensors    
        x_tensor[:, :, :, 0] = self.x[self.matrix[batch_start:batch_end, 0], :, :]
        x_tensor[:, :, :, 1] = self.x[self.matrix[batch_start:batch_end, 1], :, :]
            
        if self.output == 'x': 
            return x_tensor, x_tensor
        elif self.output == 'y':
            # Collect labels into the tensors
            y_tensor = np.zeros((batch_size, 2))
            y_tensor = self.pair_labels[batch_start:batch_end]

            return x_tensor, y_tensor[:,2]
        elif self.output == 'xy':
            # Collect labels into the tensors
            y_tensor = np.zeros((batch_size, 2))
            y_tensor = self.pair_labels[batch_start:batch_end]

            return x_tensor, {'recon': x_tensor, 'mean-var': np.zeros(len(x_tensor)), 'distance': y_tensor[:,2]}
            
        else:
            raise RuntimeError('Unexpected output format.')

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.matrix) // self.batch_size

    def on_epoch_end(self):
        # Make new triplet indexes at the end of each epoch
        if self.can_shuffle:
            self.make_pair_matrix()