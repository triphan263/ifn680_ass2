'''

2017 IFN680 Assignment Two

Scaffholding code to get you started for the 2nd assignment.


'''

import random
import numpy as np

#import matplotlib.pyplot as plt

from tensorflow.contrib import keras

from tensorflow.contrib.keras import backend as K

from tensorflow.contrib.keras.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.contrib.keras.api.keras.models import load_model

from sklearn.metrics import roc_curve, auc, roc_auc_score

import matplotlib.pyplot as plt

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
            # z1 and z2 form a positive pair
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            # z1 and z2 form a negative pair
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

#------------------------------------------------------------------------------
    
def simplistic_solution():
    '''
    
    Train a Siamese network to predict whether two input images correspond to the 
    same digit.
    
    WARNING: 
        in your submission, you should use auxiliary functions to create the 
        Siamese network, to train it, and to compute its performance.
    
    
    '''
    def create_simplistic_base_network(input_dim):
        '''
        Base network to be shared (eq. to feature extraction).
        '''
        seq = keras.models.Sequential()
        seq.add(keras.layers.Dense(128, input_shape=(input_dim,), activation='relu'))
        seq.add(keras.layers.Dropout(0.1))
        seq.add(keras.layers.Dense(128, activation='relu'))
        seq.add(keras.layers.Dropout(0.1))
        seq.add(keras.layers.Dense(128, activation='relu'))
        return seq
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    
    # load the dataset
    x_train, y_train, x_test, y_test  = assign2_utils.load_dataset()

    # Example of magic numbers (6000, 784)
    # This should be avoided. Here we could/should have retrieve the
    # dimensions of the arrays using the numpy ndarray method shape 
    x_train = x_train.reshape(60000, 784) 
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255 # normalized the entries between 0 and 1
    x_test /= 255
    input_dim = 784 # 28x28

    #
    epochs = 20

    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)
    
    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)
    
    # network definition
    base_network = create_simplistic_base_network(input_dim)
    
    input_a = keras.layers.Input(shape=(input_dim,))
    input_b = keras.layers.Input(shape=(input_dim,))
    
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # node to compute the distance between the two vectors
    # processed_a and processed_a
    distance = keras.layers.Lambda(euclidean_distance)([processed_a, processed_b])
    
    # Our model take as input a pair of images input_a and input_b
    # and output the Euclidian distance of the mapped inputs
    model = keras.models.Model([input_a, input_b], distance)

    callback = siamese_callback([te_pairs[:, 0], te_pairs[:, 1]], te_y)
#    filepath="weights.best.hdf5"
#    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=2, patience=0, verbose=0, mode='auto')
#    checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    callback_list = [callback]
    # train
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    # important code - donot delete line below for loading model from file
#    model = load_model('best_model.h5', custom_objects={'contrastive_loss': contrastive_loss})
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,
              epochs=epochs,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y), 
              callbacks=callback_list)
    
    # compute final accuracy on training and test sets
    pred_train = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred_train, tr_y)
    pred_test = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred_test, te_y)
#    
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    visualise_roc(pred_train, tr_y, pred_test, te_y)
#------------------------------------------------------------------------------
class siamese_callback(keras.callbacks.Callback):
    def __init__(self, te_pairs, te_y):
        self.te_pairs = te_pairs
        self.te_y = te_y
    def on_epoch_end(self, epoch, logs={}):
        pred_train = self.model.predict(self.te_pairs, verbose=0)
        tr_acc = compute_accuracy(pred_train, self.te_y)
        print('\nAccuracy on test set: %0.2f%%' % (100 * tr_acc))
#        # important code - do not delete line below for saving model to file
#        if (tr_acc > 0.85):
#            self.model.save('best_model.h5')
#            self.model.stop_training = True
#            print('model saved')
#            print('model training stopped')

def visualise_roc(pred_train, tr_y, y_pred, y_test):
#    fpr_tr, tpr_tr, thresholds_tr = roc_curve(tr_y, pred_train, pos_label = 0)
    fpr_te, tpr_te, thresholds_te = roc_curve(y_test, y_pred, pos_label = 0)

#    plt.figure()
#    lw = 2
#    plt.plot(fpr_tr, tpr_tr, color='darkorange',
#             lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr_tr, tpr_tr))
#    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('ROC on training set')
#    plt.legend(loc="lower right")
    
    plt.figure()
    lw = 2
    plt.plot(fpr_te, tpr_te, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr_te, tpr_te))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on test set')
    plt.legend(loc="lower right")
    
    plt.show()

#------------------------------------------------------------------------------        
#------------------------------------------------------------------------------        

if __name__=='__main__':
    simplistic_solution()
    
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               CODE CEMETARY        
    
