'''

2017 IFN680 Assignment Two

Scaffholding code to get you started for the 2nd assignment.


'''
import os
import random
import numpy as np

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.contrib import keras
from tensorflow.contrib.keras import backend as K
import assign2_utils

#------------------------------------------------------------------------------

def euclidean_distance(vects):
    '''
    Auxiliary function to compute the Euclidian distance between two vectors
    in a Keras layer.
    '''
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

#------------------------------------------------------------------------------

def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    @param
      y_true : true label 1 for positive pair, 0 for negative pair
      y_pred : distance output of the Siamese network    
    '''
    margin = 1
    # if positive pair, y_true is 1, penalize for large distance returned by Siamese network
    # if negative pair, y_true is 0, penalize for distance smaller than the margin
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
#------------------------------------------------------------------------------

def compute_accuracy(predictions, labels):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    @param 
      predictions : values computed by the Siamese network
      labels : 1 for positive pair, 0 otherwise
    '''
    # the formula below, compute only the true positive rate]
    #    return labels[predictions.ravel() < 0.5].mean()
    n = labels.shape[0]
    acc =  (labels[predictions.ravel() < 0.5].sum() +  # count True Positive
               (1-labels[predictions.ravel() >= 0.5]).sum() ) / n  # True Negative
    return acc

#------------------------------------------------------------------------------

def create_pairs(x, digit_indices):
    '''
       Positive and negative pair creation.
       Alternates between positive and negative pairs.
       @param
         digit_indices : list of lists
            digit_indices[k] is the list of indices of occurences digit k in 
            the dataset
       @return
         P, L 
         where P is an array of pairs and L an array of labels
         L[i] ==1 if P[i] is a positive pair
         L[i] ==0 if P[i] is a negative pair
         
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

#------------------------------------------------------------------------------
    
def siamese_solution(x_train, y_train, x_test, y_test, epochs):
    '''
    Train a Siamese network to predict whether two input images correspond to the 
    same digit.
    
    WARNING: 
        in your submission, you should use auxiliary functions to create the 
        Siamese network, to train it, and to compute its performance.
    
    
    '''
    def create_siamese_convolutional_network(input_shape):
        '''
        Base network to be shared (eq. to feature extraction).
        '''
        seq = keras.models.Sequential()
        seq.add(keras.layers.Conv2D(30, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        seq.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        seq.add(keras.layers.Conv2D(15, (3, 3), activation='relu'))
        seq.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        seq.add(keras.layers.Dropout(0.2))
        seq.add(keras.layers.Flatten())
        seq.add(keras.layers.Dense(128, activation='relu'))
        seq.add(keras.layers.Dense(50, activation='relu'))
        seq.add(keras.layers.Dense(10, activation='softmax'))
        
#        seq = keras.models.Sequential()
#        seq.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
#        seq.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#        seq.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
#        seq.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#        seq.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
#        seq.add(keras.layers.Flatten())
#        seq.add(keras.layers.Dense(128, activation='relu'))
#        seq.add(keras.layers.Dropout(0.1))
#        seq.add(keras.layers.Dense(10, activation='softmax'))
    
        return seq
    #--------------------------------------------------------------------------
    
    def train_network(tr_pairs, tr_y, te_pairs, te_y, input_shape):
        base_network = create_siamese_convolutional_network(input_shape)
        
        input_a = keras.layers.Input(shape=input_shape)
        input_b = keras.layers.Input(shape=input_shape)
        
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        distance = keras.layers.Lambda(euclidean_distance)([processed_a, processed_b])
        
        model = keras.models.Model([input_a, input_b], distance)
        
        rms = keras.optimizers.RMSprop()
        model.compile(loss=contrastive_loss, optimizer=rms)
        model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=128,
                  epochs=epochs,
                  validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
        
        pred_train = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
        pred_test = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
        
        return pred_train, tr_y, pred_test, te_y
    #--------------------------------------------------------------------------
    
    def evaluate_accuracy(pred_train, tr_y, pred_test, te_y):
        tr_acc = compute_accuracy(pred_train, tr_y)
        te_acc = compute_accuracy(pred_test, te_y)
        
        print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    #--------------------------------------------------------------------------
    
    def receiver_operating_characteristic_curve(pred_test, te_y):
        
        fpr, tpr, thresholds = roc_curve(te_y, pred_test, pos_label = 0)
        
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC on test set')
        plt.legend(loc='lower right')
        plt.show()
    #--------------------------------------------------------------------------
    
    img_rows, img_cols = x_train.shape[1:3]
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)
    
    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)
    
    pred_train, tr_y, pred_test, te_y = train_network(tr_pairs, tr_y, te_pairs, te_y, input_shape)
    
    evaluate_accuracy(pred_train, tr_y, pred_test, te_y)
    
    # show roc chart
    receiver_operating_characteristic_curve(pred_test, te_y)

#------------------------------------------------------------------------------
#                           IMPORT/MODIFY DATASET
class SiameseExperiment:
    '''
    Collections of various methods for running the experiment based on Methodology
    
    '''
    def __init__():
        pass
    
    @staticmethod
    def run_experiments(dataset_generation, network_configs):
        '''
        Execute experiment given the settings
        Args:
            dataset_generation: something here
            network_configs: something here
        '''
        initialize_dataset(dataset_generation)
        
        dataset = 'original dataset' if network_configs['original_dataset'] else 'warped dataset'
        print('Training on {0}'.format(dataset))
        
#        x_train, y_train, x_test, y_test = load_original_data() \
#            if network_configs['original_dataset'] else load_warped_dataset()
        if network_configs['train_dataset'] == 'O'
            #Train and test on Oringinal data
            x_train, y_train, x_test, y_test = load_original_data()
            train_network(x_train, y_train, x_test, y_test, network_configs['epochs'])
            siamese_solution(x_train, y_train, x_test, y_test, network_configs['epochs'])
        elif network_configs['train_dataset'] == 'EH'
            #First tain on easy then hard and then run the test 
            x_train, y_train, x_test, y_test = load_warped_dataset('E')
            train_network(x_train, y_train, x_test, y_test, network_configs['epochs'])
            x_train, y_train, x_test, y_test = load_warped_dataset('H')
            train_network(x_train, y_train, x_test, y_test, network_configs['epochs'])
            siamese_solution(x_train, y_train, x_test, y_test, network_configs['epochs'])
        elif network_configs['train_dataset'] == 'H'
            #Train and test on hard dataset (same size of E+H)
            x_train, y_train, x_test, y_test = load_warped_dataset('HH')
            train_network(x_train, y_train, x_test, y_test, network_configs['epochs'])
            siamese_solution(x_train, y_train, x_test, y_test, network_configs['epochs'])
        else:
            #Try all the scenarios together in one go
            x_train, y_train, x_test, y_test = load_original_data()
            train_network(x_train, y_train, x_test, y_test, network_configs['epochs'])
            siamese_solution(x_train, y_train, x_test, y_test, network_configs['epochs'])
            
            x_train, y_train, x_test, y_test = load_warped_dataset('E')
            train_network(x_train, y_train, x_test, y_test, network_configs['epochs'])
            x_train, y_train, x_test, y_test = load_warped_dataset('H')
            train_network(x_train, y_train, x_test, y_test, network_configs['epochs'])
            siamese_solution(x_train, y_train, x_test, y_test, network_configs['epochs'])

            x_train, y_train, x_test, y_test = load_warped_dataset('HH')
            train_network(x_train, y_train, x_test, y_test, network_configs['epochs'])
            siamese_solution(x_train, y_train, x_test, y_test, network_configs['epochs'])

        
#------------------------------------------------------------------------------ 
#                           IMPORT/MODIFY DATASET

def initialize_dataset(dataset):
    '''
    Generate datasets and save them into file. 
    If there is not physical file, the dataset always be generated
        Arg:
            dataset: the dataset in SETTINGS which is SETTINGS['dataset']
    
    '''
    if not os.path.isfile('./mnist_dataset.npz') or dataset['create_original_dataset']:
        x_train, y_train, x_test, y_test = assign2_utils.load_dataset()
        np.savez('mnist_dataset.npz', x_train=x_train, y_train=y_train,
                                  x_test=x_test, y_test=y_test)
    
    if not os.path.isfile('./mnist_warped_dataset.npz') or dataset['create_warped_dataset']:
        generate_warped_dataset(dataset)
        
def generate_warped_dataset(dataset):
    '''
    Generate warped dataset based on parameters from SETTINGS.
    This will generate images randomly from x_train of mnist dataset
    and generate test images randomly from x_test of mnist dataset
        Args:
            dataset['warped_dataset_records']: total warped images will be generated
            dataset['hard_pair']: proportion of significant deformation of images in the dataset
                                  for example, dataset['hard_pair'] = 0.6 means 60% of warped 
                                  images in the dataset are deformed significantly
            dataset['test_set']: proportion of test set.
                                 for example, dataset['test_set'] = 0.1 means the number of
                                 test images will be 10% of warped images.
                                 dataset['test_set'] = 0.1 and 
                                 dataset['warped_dataset_records']= 100000
                                 There are 10000 test images and 100000 warped images
                                 Total is 110000 images
            
    '''
    x_train, y_train, x_test, y_test = load_original_data()
    
    # hard_pair is percentage of H dataset in total warped dataset
    hard_pair_records = int(dataset['hard_pair'] * dataset['warped_dataset_records'])
    easy_pair_records = int((1 - dataset['hard_pair']) * dataset['warped_dataset_records'])
    test_records = int(dataset['test_set'] * dataset['warped_dataset_records'])
    
    print('Warped images: ' + str(dataset['warped_dataset_records']))
    
    if network_configs['train_dataset'] == 'EH'
        h_x_train = np.zeros((hard_pair_records, x_train.shape[1], x_train.shape[2]))
        h_y_train = np.zeros((hard_pair_records))
        h_x_test = np.zeros((int(dataset['hard_pair'] * test_records), x_train.shape[1], x_train.shape[2]))
        h_y_test = np.zeros((int(dataset['hard_pair'] * test_records)))
        for i in range(hard_pair_records):
            index = np.random.randint(0, x_train.shape[0])
            h_x_train[i] = assign2_utils.random_deform(#x_train[index], 
                                 x_train[index]/255,
                                 dataset['hard_pair_rotation'], 
                                 dataset['hard_pair_variation'])
            h_y_train[i] = y_train[index]
        for i in range(int(dataset['hard_pair'] * test_records)):
            index = np.random.randint(0, x_test.shape[0])
            h_x_test[i] = assign2_utils.random_deform(#x_test[index], 
                                 x_test[index]/255,
                                 dataset['hard_pair_rotation'], 
                                 dataset['hard_pair_variation'])
            h_y_test[i] = y_test[index]
            
        print('Hard (H) dataset: ' + str(h_x_train.shape))
        
        e_x_train = np.zeros((easy_pair_records, x_train.shape[1], x_train.shape[2]))
        e_y_train = np.zeros((easy_pair_records))
        e_x_test = np.zeros((int((1 - dataset['hard_pair']) * test_records), x_train.shape[1], x_train.shape[2]))
        e_y_test = np.zeros((int((1 - dataset['hard_pair']) * test_records)))
        for i in range(easy_pair_records):
            index = np.random.randint(0, x_train.shape[0])
            e_x_train[i] = assign2_utils.random_deform(#x_train[index], #
                                 x_train[index]/255,
                                 dataset['easy_pair_rotation'], 
                                 dataset['easy_pair_variation'])
            e_y_train[i] = y_train[i]
        for i in range(int((1 - dataset['hard_pair']) * test_records)):
            index = np.random.randint(0, x_test.shape[0])
            e_x_test[i] = assign2_utils.random_deform(#x_test[index], # 
                                 x_test[index]/255,
                                 dataset['easy_pair_rotation'], 
                                 dataset['easy_pair_variation'])
            e_y_test[i] = y_test[i]
    elif network_configs['train_dataset'] == 'H'
        #Warp the daataset with H setting same size of E+H 
    print('Easy (E) dataset: ' + str(e_x_train.shape))
    
    warped_x_train = np.concatenate((h_x_train, e_x_train), axis=0)
    warped_y_train = np.concatenate((h_y_train, e_y_train), axis=0)
    warped_x_test  = np.concatenate((h_x_test, e_x_test), axis=0)
    warped_y_test = np.concatenate((h_y_test, e_y_test), axis=0)    
    
    print('Test dataset: ' + str(warped_x_test.shape))
    np.savez('mnist_warped_dataset.npz', x_train=warped_x_train, y_train=warped_y_train,
             x_test=warped_x_test, y_test=warped_y_test)
    
def load_original_data():
    '''
    Load the dataset, shuffled and split between train and test sets
    and return the numpy arrays  x_train, y_train, x_test, y_test in original dataset
    The dtype of all returned array is uint8
    
    '''
    assert os.path.isfile('./mnist_dataset.npz')
    with np.load('mnist_dataset.npz') as npzfile:
#        x_train = npzfile['x_train']/255
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']
#        x_test = npzfile['x_test']/255
        x_test = npzfile['x_test']
        y_test = npzfile['y_test']
    
    return x_train, y_train, x_test, y_test

def load_warped_dataset(dataset_type):
    '''
    Load the dataset, shuffled and split between train and test sets
    and return the numpy arrays  x_train, y_train, x_test, y_test in warped dataset
    The dtype of all returned array is uint8
    
    '''
    if dataset_type == 'E'
        assert os.path.isfile('./mnist_easy_warped_dataset.npz')
        datasetname = './mnist_easy_warped_dataset.npz'
    if dataset_type == 'H'
        assert os.path.isfile('./mnist_partialhard_warped_dataset.npz')
        datasetname = './mnist_partialhard_warped_dataset.npz'
    if dataset_type == 'HH'
        assert os.path.isfile('./mnist_fullhard_warped_dataset.npz')
        datasetname = './mnist_fullhard_warped_dataset.npz'
    with np.load(datasetname) as npzfile:
        warped_x_train = npzfile['x_train']
        warped_y_train = npzfile['y_train']
        warped_x_test = npzfile['x_test']
        warped_y_test = npzfile['y_test']
    
    return warped_x_train, warped_y_train, warped_x_test, warped_y_test


#------------------------------------------------------------------------------
# CONFIGURATION AND SETTING

SETTINGS = {
    'dataset_generation': {
        'create_original_dataset': False,   # Donwload the dataset and save to the file named
                                            # mnist_dataset.npz. By doing this, we don't have
                                            # to download the dataset every run the experiment
                                            
        'create_warped_dataset': True,      # Using mnist_dataset.npz, we generate warped dataset
                                            # and save the dataset to mnist_warped_dataset.npz
                                            # When we want to generate warped dataset again,
                                            # we just need to set it to True. 
                                            
        'warped_dataset_records': 100000,   # number of records in warped dataset (D dataset)
        'hard_pair': 0.8,                   # in percentage from 0 to 1. for example, 
                                            # if hard_pair=0.6 and warped_dataset_records=100000,  
                                            # the hard pairs will be 60% of warped_dataset_records which is
                                            # 60000 and the easy_pair will be 40000
                                            
        'test_set': 0.1,                    # proportion of test dataset.  
                                            # test_set = 0.2 means extra 20% of the warped 
                                            # images in the dataset will be used for test dataset
                                            
        'hard_pair_rotation': 45,           # rotation of deformed the dataset (H dataset)
        'hard_pair_variation': 0.3,         # rotation of deformed the dataset (H dataset)
        
        'easy_pair_rotation': 5,            # rotation of deformed the dataset (E dataset)
        'easy_pair_variation': 0.01,        # variation of deformed the dataset (E dataset)

    },
    'network_configs': {                    # experiment on original dataset
        'epochs': 2,
        'train_dataset': 'O'                # original_dataset can take values from ['O', 'H', 'EH', A]. 
                                            # O: Using original dataset
                                            # H: Using Hard dataset equal to the size of E+H
                                            # EH: Using Easy and then Hard dataset
                                            # to train the network
    }
}
    
#------------------------------------------------------------------------------        

if __name__=='__main__':
    SiameseExperiment.run_experiments(**SETTINGS)