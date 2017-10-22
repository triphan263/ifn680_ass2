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
    
def siamese_solution(x_train, y_train, x_test, y_test, epochs, retrain, 
                     threshold_acc, model_filename):
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
        
        # DO NOT DELETE THIS IMPORTANT
        seq = keras.models.Sequential()
        seq.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        seq.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        seq.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        seq.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        seq.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        seq.add(keras.layers.Flatten())
        seq.add(keras.layers.Dense(128, activation='relu'))
        seq.add(keras.layers.Dropout(0.1))
        seq.add(keras.layers.Dense(128, activation='relu'))
    
        # using very simple base network for testing only
        #seq = keras.models.Sequential()
        #seq.add(keras.layers.Dense(128, input_shape=(784,), activation='relu'))
        #seq.add(keras.layers.Dropout(0.1))
        #seq.add(keras.layers.Dense(128, activation='relu'))
        #seq.add(keras.layers.Dropout(0.1))
        #seq.add(keras.layers.Dense(128, activation='relu'))
        
        return seq
    #--------------------------------------------------------------------------
    
    def train_network(tr_pairs, tr_y, te_pairs, te_y, input_shape):
        
        if retrain and os.path.exists(model_filename):
            print('Retraining network from {0}'.format(model_filename))
            model = keras.models.load_model(model_filename, 
                                            custom_objects={'contrastive_loss': contrastive_loss})
        else:
            print('Start training new network')
            base_network = create_siamese_convolutional_network(input_shape)
            
            # DO NOT DELETE THIS IMPORTANT
            input_a = keras.layers.Input(shape=input_shape)
            input_b = keras.layers.Input(shape=input_shape)

            # using very simple base network for testing only
#            input_a = keras.layers.Input(shape=(784,))
#            input_b = keras.layers.Input(shape=(784,))
            
            processed_a = base_network(input_a)
            processed_b = base_network(input_b)
            
            distance = keras.layers.Lambda(euclidean_distance)([processed_a, processed_b])
            
            model = keras.models.Model([input_a, input_b], distance)
            
            rms = keras.optimizers.RMSprop()
            model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
        
        callback = siamese_callback([te_pairs[:, 0], te_pairs[:, 1]], te_y,
                                    threshold_acc, model_filename)
        callback_list = [callback]
        
        model_info =  model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=128,
                  epochs=epochs,
                  validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
                  callbacks=callback_list)
        
        pred_train = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
        pred_test = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
        
        return pred_train, tr_y, pred_test, te_y, model_info
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
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC on test set')
        plt.legend(loc='lower right')
        plt.show()
    #--------------------------------------------------------------------------
        
    def plot_model_history(model_history):
        plt.figure()
        # summarize history for loss
        plt.plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
        plt.plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
#        plt.xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
        plt.legend(['train', 'test'], loc='best')
        plt.show()
    #--------------------------------------------------------------------------
    
    # DO NOT DELETE THIS IMPORTANT
    img_rows, img_cols = x_train.shape[1:3]
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # using very simple base network for testing only
#    x_train = x_train.reshape(x_train.shape[0], 784) 
#    x_test = x_test.reshape(x_test.shape[0], 784)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)
    
    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)
    
    pred_train, tr_y, pred_test, te_y, model_info = train_network(tr_pairs, tr_y, te_pairs, te_y, input_shape)
    #pred_train, tr_y, pred_test, te_y, model_info = train_network(tr_pairs, tr_y, te_pairs, te_y, 784)
    
    #evaluate_accuracy(pred_train, tr_y, pred_test, te_y)
    #plot_model_history(model_info)
    # show roc chart
    receiver_operating_characteristic_curve(pred_test, te_y)


#------------------------------------------------------------------------------
class siamese_callback(keras.callbacks.Callback):
    def __init__(self, te_pairs, te_y, threshold_acc, model_filename):
        self.te_pairs = te_pairs
        self.te_y = te_y
        self.threshold_acc = threshold_acc
        self.model_filename = model_filename
        self.losses = []
        
    def on_train_end(self, logs={}):
        pred_train = self.model.predict(self.te_pairs, verbose=0)
        tr_acc = compute_accuracy(pred_train, self.te_y)
        if (tr_acc < self.threshold_acc):
            self.model.save(self.model_filename)            
            print('Model ended. A file of the model has been saved into {0}'.format(
                    self.model_filename))
    
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        
        pred_train = self.model.predict(self.te_pairs, verbose=0)
        tr_acc = compute_accuracy(pred_train, self.te_y)
        print('\nAccuracy on test set: %0.2f%%' % (100 * tr_acc))
        # important code - do not delete line below for saving model to file
        if (tr_acc > self.threshold_acc):
            self.model.save(self.model_filename)            
            print('Model saved into {0}'.format(self.model_filename))
            self.model.stop_training = True
            print('Model stopped. You can retrain the model')

#------------------------------------------------------------------------------
#                           Operation experiment
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
        
        if (network_configs['train_dataset'] == 'O'):
            x_train, y_train, x_test, y_test = load_original_data(dataset_generation['original_file_name'])
            siamese_solution(x_train, y_train, x_test, y_test, network_configs['epochs'], 
                             network_configs['retrain'], network_configs['threshold_acc'],
                             network_configs['model_filename'])
        else:            
            x_train, y_train = load_warped_dataset(dataset_generation['easyset_file_name']) \
                if network_configs['train_dataset'] == 'E' else \
                load_warped_dataset(dataset_generation['hardset_file_name'])
            
            x_test, y_test = load_test_data()
            
            siamese_solution(x_train, y_train, x_test, y_test, network_configs['epochs'], 
                             network_configs['retrain'], network_configs['threshold_acc'],
                             network_configs['model_filename'])
                
        print('Trained on {0} dataset.'.format(network_configs['train_dataset']))
        print('Retrain: {0}.'.format(network_configs['retrain']))
        
#------------------------------------------------------------------------------ 
#                           IMPORT/MODIFY DATASET

def initialize_dataset(dataset):
    '''
    Generate datasets and save them into file. 
    If there is not physical file, the dataset always be generated
        Arg:
            dataset: the dataset in SETTINGS which is SETTINGS['dataset']
    
    '''
    if not os.path.isfile(dataset['original_file_name']) or dataset['create_original_dataset']:
        x_train, y_train, x_test, y_test = assign2_utils.load_dataset()
        np.savez(dataset['original_file_name'], x_train=x_train, y_train=y_train,
                                  x_test=x_test, y_test=y_test)
    
    if not os.path.isfile(dataset['hardset_file_name']) or dataset['create_hard_dataset']:
        generate_warped_dataset(dataset['original_file_name'], dataset['hardset_file_name'], 
                                dataset['hard_set'], dataset['hard_pair_rotation'], 
                                dataset['hard_pair_variation'])
        
    if not os.path.isfile(dataset['easyset_file_name']) or dataset['create_easy_dataset']:
        generate_warped_dataset(dataset['original_file_name'], dataset['easyset_file_name'], 
                                dataset['easy_set'], dataset['easy_pair_rotation'], 
                                dataset['easy_pair_variation'])
    
    if not os.path.isfile('./test_dataset.npz') or dataset['create_test_dataset']:
        generate_testset(dataset['original_file_name'], dataset['test_set'],
                         dataset['testset_rotation'], dataset['testset_variation'])

def generate_warped_dataset(original_file_name, file_name, number, rotation, variation):    
    '''
        Need comment
            
    '''
    x_train, y_train, x_test, y_test = load_original_data(original_file_name)
    
    warped_x_train = np.zeros((number, x_train.shape[1], x_train.shape[2]))
    warped_y_train = np.zeros((number))
    
    for i in range(number):
        index = np.random.randint(0, x_train.shape[0])
        warped_x_train[i] = assign2_utils.random_deform(x_train[index], #[index]/255,
                             rotation, 
                             variation)
        warped_y_train[i] = y_train[index]
        
    np.savez(file_name, x_train=warped_x_train, y_train=warped_y_train)
    print('Generated file {0} with training shape: {1}'.format(file_name, warped_x_train.shape))
        
def generate_testset(original_file_name, number, rotation, variation):
    x_train, y_train, x_test, y_test = load_original_data(original_file_name)
    
    x_te = np.zeros((number, x_test.shape[1], x_test.shape[2]))
    y_te = np.zeros((number))
    
    for i in range(number):
        index = np.random.randint(0, x_test.shape[0])
        x_te[i] = assign2_utils.random_deform(x_test[index], rotation, variation)
        y_te[i] = y_train[index]
        
    np.savez('test_dataset.npz', x_test=x_te, y_test=y_te)
    print('Generated test with training shape: {0}'.format(x_te.shape))
    
def load_original_data(file_name):
    '''
    Load the dataset, shuffled and split between train and test sets
    and return the numpy arrays  x_train, y_train, x_test, y_test in original dataset
    The dtype of all returned array is uint8
    
    '''
    assert os.path.isfile(file_name)
    with np.load(file_name) as npzfile:
        x_train = npzfile['x_train']/255
        y_train = npzfile['y_train']
        x_test = npzfile['x_test']/255
        y_test = npzfile['y_test']
    
    return x_train, y_train, x_test, y_test

def load_warped_dataset(file_name):
    '''
    Load the dataset, shuffled and split between train and test sets
    and return the numpy arrays  x_train, y_train, x_test, y_test in warped dataset
    The dtype of all returned array is uint8
    
    '''
    assert os.path.isfile(file_name)
    with np.load(file_name) as npzfile:
        warped_x_train = npzfile['x_train']
        warped_y_train = npzfile['y_train']
#        warped_x_test = npzfile['x_test']
#        warped_y_test = npzfile['y_test']
    
    return warped_x_train, warped_y_train#, warped_x_test, warped_y_test

def load_test_data():
    '''
    Load the dataset, shuffled and split between train and test sets
    and return the numpy arrays  x_train, y_train, x_test, y_test in original dataset
    The dtype of all returned array is uint8
    
    '''
    assert os.path.isfile('./test_dataset.npz')
    with np.load('test_dataset.npz') as npzfile:
        x_test = npzfile['x_test']
        y_test = npzfile['y_test']
    
    return x_test, y_test

#------------------------------------------------------------------------------
# CONFIGURATION AND SETTING

SETTINGS = {
    'dataset_generation': {
        'create_original_dataset': False,   # Donwload the dataset and save to the file
                                            # By doing this, we don't have to download
                                            # the dataset every run the experiment
        'original_file_name': 'mnist_original.npz',
                                            
        'create_hard_dataset': False,       # Using original dataset, we generate warped dataset
                                            # and save the dataset to file
                                            # When we want to generate warped dataset again,
                                            # we just need to set it to True.                                             
        'hardset_file_name': 'mnist_hardset.npz',
        'create_easy_dataset': False,       # Using original dataset, we generate warped dataset
                                            # and save the dataset to file
                                            # When we want to generate warped dataset again,
                                            # we just need to set it to True.                                            
        'easyset_file_name': 'mnist_easyset.npz',
                                            
        'hard_set': 60000,                  # number of images in hard dataset (H dataset)
        'hard_pair_rotation': 45,           # rotation of deformed the dataset (H dataset)
        'hard_pair_variation': 0.3,         # rotation of deformed the dataset (H dataset)
        
        'easy_set': 40000,                  # number of images in easy set
        'easy_pair_rotation': 5,            # rotation of deformed the dataset (E dataset)
        'easy_pair_variation': 0.01,        # variation of deformed the dataset (E dataset)
        
        'create_test_dataset': False,                                    
        'test_set': 10000,                  # number of images in test set.                                            
        'testset_rotation': 45,             # rotation of deformed the dataset (E dataset)
        'testset_variation': 0.03,          # variation of deformed the dataset (E dataset)
    },
    'network_configs': { # experiment on original dataset
        'epochs': 5,
        'threshold_acc': 0.95,
        'retrain': False,                                                            
        'model_filename': 'best_model.h5',
        'train_dataset': 'E'                # original_dataset can take values from ['O', 'H', 'E']. 
                                            # O: Using original dataset
                                            # E: Using Easy dataset
                                            # H: Using Hard dataset equal 
                                            # to train the network
    }
}
    
#------------------------------------------------------------------------------        

if __name__=='__main__':
    SiameseExperiment.run_experiments(**SETTINGS)