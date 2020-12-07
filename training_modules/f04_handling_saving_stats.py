# General modules
import os
import sys
from pathlib import Path # creating directories
import numpy as np
# Plot traces
import matplotlib.pyplot as plt
# plot result
from seaborn import catplot
from pandas import DataFrame
# consts
# hyper-parameters
from training_modules.hyper_parameters import VALIDATION_SPLIT_CONST


def check_file_exists(file_path):
    """
    TODO: add description
    :param file_path:
    :return:
    """
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


# # Saves pre-defined history parameters that keras training returns. It assumes existence of validation data.
def save_history(training_model, history, seed=None, key_idx=None, att=None, valid_val=None):
    """
    TODO: add description
    :param training_model:
    :param history:
    :param seed:
    :param key_idx:
    :param att:
    :param valid_val:
    :return:
    """
    # SAVE HISTORY
    # # SAVE HISTORY: LOSS per epoch
    save_file(training_model, history.history['loss'], case=LOSS_CONST, seed=seed, key=key_idx)
    # # SAVE HISTORY: ACCURACY per epoch
    save_file(training_model, history.history['accuracy'], case=ACC_CONST, seed=seed, key=key_idx)
    if valid_val is not None:
        # # SAVE HISTORY: VAL_LOSS per epoch
        save_file(training_model, history.history['val_loss'], case=VAL_LOSS_CONST, seed=seed, key=key_idx)
        # # SAVE HISTORY: VAL_ACCURACY per epoch
        save_file(training_model, history.history['val_accuracy'], case=VAL_ACC_CONST, seed=seed, key=key_idx)

    # # SAVE HISTORY: GRAPH
    save_file(training_model, history, case=TRN_GRPH_CONST, seed=seed, key=key_idx, att=att)


# fun little convention for naming saved files
# this is relevant for the saving functionality when saving and loading arrays, models, and graphs
MODEL_CONST = 0
LOSS_CONST = 1
ACC_CONST = 2
ADV_CONST = 3
TRN_GRPH_CONST = 4
ADV_GRPH_CONST = 5
TST_ACC_CONST = 6
VAL_LOSS_CONST = 7
VAL_ACC_CONST = 8
VAL_ADV_CONST = 9
TST_ADV_CONST = 10
FIN_ACC_CONST = 11
FIN_ADV_CONST = 12
CLS_ADV_GRPH_CONST = 13

# ('/start_of_file_name', '.file_extension', '/file_folder') # The index comes from the constants above
SAVE_LIST = [('/mdl', '.h5', "/model"),
             ('/trn_lss', '.npy', "/trn_loss"),
             ('/trn_acc', '.npy', "/trn_accuracy"),
             ('/trn_adv', '.npy', "/trn_advantage"),
             ('/trn_grph', '.png', "/graphs"),
             ('/adv_grph', '.png', ""),
             ('/tst_acc', '.npy', ""),
             ('/val_lss', '.npy', "/val_loss"),
             ('/val_acc', '.npy', "/val_accuracy"),
             ('/val_adv', '.npy', "/val_advantage"),
             ('/tst_adv', '.npy', ""),
             ('/fin_adv', ".npy", ""),
             ("/cls_adv", ".png", "")]


def get_file_loc(save_loc, case):
    """
    using the SAVE_LIST we create a path to the object that needs to be saved
    :param save_loc: path to file or directory of object
    :param case: value type (example: model, training loss, validation advantage, graphs, etc...)
    :return: path to file that needs to be saved
    """
    loc = save_loc + SAVE_LIST[case][2]
    Path(loc).mkdir(parents=True, exist_ok=True)
    return loc


def get_file_path(save_loc, case, seed=None, key=None):
    """
    TODO: add description
    :param save_loc:
    :param case:
    :param seed:
    :param key:
    :return:
    """
    tmp_str = get_file_loc(save_loc, case) + SAVE_LIST[case][0]
    
    if seed is not None:
        tmp_str = tmp_str + "_s{:04d}".format(seed)
    
    if key is not None:
        tmp_str = tmp_str + "_k{:02d}".format(key)
    return tmp_str + "{}".format(SAVE_LIST[case][1])


def save_file(save_loc, file, case=-1, seed=None, key=None, att=None):
    """
    TODO: add description
    :param save_loc:
    :param file:
    :param case:
    :param seed:
    :param key:
    :param att:
    :return:
    """
    
    file_path = get_file_path(save_loc, case, seed, key)
    print("++ Saving: ", file_path)

    if case in [LOSS_CONST, ACC_CONST, VAL_LOSS_CONST, VAL_ACC_CONST, VAL_ADV_CONST]:
        # This means that file is a numpy array
        np.savetxt(file_path, file)
    
    elif case in [TST_ADV_CONST, TST_ACC_CONST, FIN_ADV_CONST, FIN_ACC_CONST]:
        # This means that file is a dictionary
        np.save(file_path, file)

    elif case in [TRN_GRPH_CONST]:
        # This means we want to save a graph of a singular key
        # The graph needs the accuracy,  arrays
        # File is history
        plt.plot(file.history['accuracy'])
        plt.plot(file.history['loss'])
        if VALIDATION_SPLIT_CONST is not None:
            plt.plot(file.history['val_accuracy'])
            plt.plot(file.history['val_loss'])
        tmp_title = 'Training Graph: ' + 'seed' + '{:02d}'.format(seed)
        if key is not None:
            tmp_title = tmp_title + ", key: " + '{:02d}'.format(key)
        if att is not None and att != 0:
            tmp_title = tmp_title + ', att: ' + '{:01d}'.format(att)
        plt.title(tmp_title)
        plt.ylabel('%')
        plt.xlabel('Epoch')
        if VALIDATION_SPLIT_CONST is not None:
            plt.legend(['Acc', 'Valid. Acc', 'Loss', 'Valid. Loss'], loc='upper left')
        else:
            plt.legend(['Acc', 'Loss'], loc='upper left')
        plt.savefig(file_path)
        plt.clf()
        
    elif case in [ADV_GRPH_CONST]:
        # This means we want to save a graph of a singular key
        # The graph needs the accuracy,  arrays
        # File is history
        plt.plot(file.history['trn_advantage'])
        if VALIDATION_SPLIT_CONST is not None:
            plt.plot(file.history['val_advantage'])
        
        tmp_title = 'Advantage Graph: '
        if seed is not None:
            tmp_title = tmp_title + 'seed' + '{:02d}, '.format(seed)
        if key is not None:
            tmp_title = tmp_title + "key: " + '{:02d}, '.format(key)
        if att is not None and att != 0:
            tmp_title = tmp_title + ', att: ' + '{:01d}'.format(att)
            
        plt.title(tmp_title)
        plt.ylabel('%')
        plt.xlabel('Epoch')
        if VALIDATION_SPLIT_CONST is not None:
            plt.legend(['trn. Adv', 'val. Adv'], loc='upper left')
        else:
            plt.legend(['trn. Adv'], loc='upper left')
        plt.savefig(file_path)
        plt.clf()
    else:
        raise ValueError("Error: save_file was called with a wrong case")


# Load attacking traces
# TODO: fix this method up and make it work for ASCAD too.
def display_results(models, accuracies, advantages, training_model, data_key_num:int):
    """
    TODO: add description
    :param models:
    :param accuracies:
    :param advantages:
    :param training_model:
    :param data_key_num:
    :return:
    """

    # TODO: assert data_key_num is an instance of a positive int

    # if dataType not in [TYPE_ASCAD]:
    #     adv = {i: (seed, key_idx, (advantage[(seed, key_idx)])) for i, (seed, key_idx) in enumerate(models.keys())}
    #     print("adv: ", adv)
    #     data = DataFrame.from_dict(adv, orient='index', columns=['seed', 'key_idx', 'adv'])
    #     catplot(data=data, x="key_idx", y="adv", row="seed", kind="bar")
    #
    #     plt.savefig(get_file_path(training_model, ADV_GRPH_CONST))
    #
    # else:
    #     adv = {i: (seed, (advantage[seed])) for i, (seed) in enumerate(models.keys())}
    #     print("adv: ", adv)
    #     data = DataFrame.from_dict(adv, orient='index', columns=['seed', 'adv'])
    #     catplot(data=data, y="adv", x="seed", kind="bar")
    #     plt.savefig(get_file_path(training_model, ADV_GRPH_CONST))

    # if data_key_num == 1:
    #     adv = {i: (seed, (advantages[(seed, 0)])) for i, (seed, key_idx) in enumerate(models.keys())}
    #     print("adv: ", adv)
    #     data = DataFrame.from_dict(adv, orient='index', columns=['seed', 'key_idx', 'adv'])
    #     catplot(data=data, x="key_idx", y="adv", row="seed", kind="bar")

    #     plt.savefig(get_file_path(training_model, ADV_GRPH_CONST))

    # elif data_key_num > 1:
        # # moved the following few lines a tab back
    my_keys = models.keys()
    adv = {(seed, key_idx): (key_idx ,advantages[(seed, key_idx)]) for i, (seed, key_idx) in enumerate(models.keys())}
    print("adv: ", adv)
    data = DataFrame.from_dict(adv, orient='index', columns=["key_idx", 'adv'])
    catplot(data=data, y="adv", x="key_idx", kind="bar")
    plt.savefig(get_file_path(training_model, ADV_GRPH_CONST))

    # else:
    #     raise ValueError("Error: dataKeyNum wasn't passed")

