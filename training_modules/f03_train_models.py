# General modules
import os

# hyper parameters
from training_modules.hyper_parameters import MIN_ACC, drop_out, MAX_ATTEMPTS_PER_KEY, VALIDATION_SPLIT_CONST


# from training_modules.load_data import DATA_LOADER

from training_modules.f04_handling_saving_stats import get_file_loc, get_file_path, check_file_exists
from training_modules.f04_handling_saving_stats import MODEL_CONST

# Training methods
from keras.callbacks import ModelCheckpoint, EarlyStopping

# ASCAD code adapted by Mahmoud Gharra to fit our purposes.


# # Training high level function
def train_model(data, model, save_loc, epochs=200, batch_size=100, seed=0, key_idx=None):
    """
    TODO: add description
    :param data:
    :param model:
    :param save_loc:
    :param epochs:
    :param batch_size:
    :param seed:
    :param key_idx:
    :return:
    """
    for attempt in range(MAX_ATTEMPTS_PER_KEY):
        model_dir = get_file_loc(save_loc, MODEL_CONST)
        check_file_exists(os.path.dirname(model_dir))

        # delete pre-existing weights
        if os.path.exists(os.path.normpath(get_file_path(save_loc, MODEL_CONST, seed, key_idx))):
            os.remove(get_file_path(save_loc, MODEL_CONST, seed, key_idx))

        # Save model every epoch
        save_model = ModelCheckpoint(get_file_path(save_loc, MODEL_CONST, seed, key_idx))
        callbacks = [save_model]

        # TODO: (cont.) Look into that instead of passing data.X_profiling directly into training_model_intern
        _history = training_model_intern(model=model, x=data.X_profiling, y=data.Y_profiling, callbacks=callbacks,
                                         batch_size=batch_size, verbose=1, epochs=epochs,
                                         validation_split=VALIDATION_SPLIT_CONST)
        
        if _history.history['accuracy'][-1] > MIN_ACC:
            break
        # getting here means we probably want to retry the training with new values
    # TODO: instead of resetting _history after each attempt, add it to the final history
    return _history, attempt+1


def training_model_intern(model, x, y, callbacks, batch_size=100, verbose=1, epochs=150, validation_split=0.1):
    """
    TODO: add description
    :param model:
    :param x:
    :param y:
    :param callbacks:
    :param batch_size:
    :param verbose:
    :param epochs:
    :param validation_split:
    :return:
    """

    if validation_split is not None:
        _history = model.fit(x=x, y=y, batch_size=batch_size, verbose=verbose, epochs=epochs, callbacks=callbacks,
                             validation_split=validation_split)
    else:
        _history = model.fit(x=x, y=y, batch_size=batch_size, verbose=verbose, epochs=epochs, callbacks=callbacks)
    return _history
