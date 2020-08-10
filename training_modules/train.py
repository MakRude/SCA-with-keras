# General modules
import os

# hyper parameters
from hyper_parameters import epochs, batch_size, MIN_ACC, drop_out, MAX_ATTEMPTS_PER_KEY, LEARNING_RATE, validation_split_const


from training_modules.load_data import DATA_LOADER

from training_modules.handling_saving_stats import get_file_loc, get_file_path
from training_modules.handling_saving_stats import MODEL_CONST

from training_modules.misc import check_file_exists

# Training modules
## CNN
### Model modules
from keras.layers import Input, Conv1D, AveragePooling1D, Flatten, Dense, AveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
### training method modules
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf
import random as python_random

## CNN2
### Model modules
from keras.layers import BatchNormalization, GaussianNoise, MaxPooling1D, Dropout

## MLP
### Model modules
from keras.optimizers import RMSprop
from keras.models import Sequential

# ASCAD code adapted by Mahmoud Gharra to fit our purposes.


## Training high level function 
def train_model(data_loader, save_loc, seed=0, key_idx=None):

    for attempt in range(MAX_ATTEMPTS_PER_KEY):
        model_dir = get_file_loc(save_loc, MODEL_CONST)
        check_file_exists(os.path.dirname(model_dir))
        # delete pre-existing weights
        if (os.path.normpath(get_file_path(save_loc, MODEL_CONST, seed, key_idx)) == True):
            os.remove(get_file_path(save_loc, MODEL_CONST, seed, key_idx))
        # Save model every epoch
        save_model = ModelCheckpoint(get_file_path(save_loc, MODEL_CONST, seed, key_idx))
        callbacks = [save_model]

        # Get the input layer shape
        input_layer_shape = data_loader.model.get_layer(index=0).input_shape
#         # Sanity check
#         if input_layer_shape[1] != len(data_loader.X_profiling[0]):
#             print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
#             sys.exit(-1)
            
            
#         # Adapt the data shape according our model input
#         if len(input_layer_shape) == 2:
#             # This is a MLP
#             Reshaped_X_profiling = data_loader.X_profiling
#         elif len(input_layer_shape) == 3:
#             # This is a CNN: expand the dimensions
#             Reshaped_X_profiling = data_loader.X_profiling.reshape((data_loader.X_profiling.shape[0], data_loader.X_profiling.shape[1], 1))
#         else:
#             print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
#             sys.exit(-1)

        Reshaped_X_profiling = data_loader.X_profiling
#         if not data_loader.TO_CAT:
        _history = training_model_intern(model=data_loader.model, x=Reshaped_X_profiling, y=data_loader.Y_profiling, callbacks=callbacks, batch_size=batch_size, verbose=1, epochs=epochs, validation_split=validation_split_const)
#         else:
#             _history = training_model_intern(model=data_loader.model, x=Reshaped_X_profiling, y=to_categorical(data_loader.Y_profiling, num_classes=data_loader.num_classes), callbacks=callbacks, batch_size=batch_size, verbose=1, epochs=epochs, validation_split=validation_split_const)
        
        if _history.history['accuracy'][-1] > MIN_ACC:
            break
        # getting here means we probably want to retry the training with new values
    # TODO: instead of resetting _history after each attempt, add it to the final history
    return _history, attempt+1


def training_model_intern(model, x, y, callbacks, batch_size=100, verbose=1, epochs=150, validation_split=0.1):
    if validation_split is not None:
        _history = model.fit(x=x, y=y, batch_size=batch_size, verbose = verbose, epochs=epochs, callbacks=callbacks, validation_split=0.1)
    else:
        _history = model.fit(x=x, y=y, batch_size=batch_size, verbose = verbose, epochs=epochs, callbacks=callbacks)
    return _history
