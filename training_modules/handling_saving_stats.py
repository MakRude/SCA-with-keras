# General modules
from pathlib import Path # creating directories
import numpy as np
# Plot traces
from scipy import signal, fftpack
import matplotlib.pyplot as plt
# plot result
from seaborn import catplot
from pandas import DataFrame
# consts
from training_modules.misc import TYPE_ASCAD, TYPE_NTRU, TYPE_GAUSS, TYPE_DPA, TYPE_M4SC
from training_modules.misc import MODEL_CONST, LOSS_CONST, ACC_CONST, ADV_CONST, TRN_GRPH_CONST, ADV_GRPH_CONST, TST_ACC_CONST, VAL_LOSS_CONST, VAL_ACC_CONST, VAL_ADV_CONST, TST_ADV_CONST
# hyper-parameters
from hyper_parameters import validation_split_const


# This method was taken from the ASCAD code and adapted very heavily

def calc_tst_acc(data_loader, seed=None, key_idx=None):
    best_model = data_loader.model
    X_attack = data_loader.X_testing
    Y_attack = data_loader.Y_testing
    TEST_NUM = data_loader.TEST_NUM
    
    if key_idx is not None:
        tmp_output = "++ calculating accuracy for seed {} and key {}".format(seed,key_idx)
    else:
        tmp_output =  "++ calculating accuracy for seed {}".format(seed)
    print(tmp_output)
    
    # Get the input layer shape
    input_layer_shape = best_model.get_layer(index=0).input_shape
    
#     # Adapt the data shape according our model input
#     if len(input_layer_shape) == 2:
#         # This is a MLP
#         Reshaped_X_attack = X_attack
#     elif len(input_layer_shape) == 3:
#         # This is a CNN: expand the dimensions
#         Reshaped_X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
#     else:
#         print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
#         sys.exit(-1)
    
    Reshaped_X_attack = X_attack
    
    predictions = np.argmax(best_model.predict(Reshaped_X_attack), 1)
#     print("predictions: ", predictions)
#     print("Y_profiling: ", Y_attack)
#     print("predictions cmp Y_profiling: ", (Y_attack == predictions))
    
    accuracy = sum(Y_attack == predictions)/TEST_NUM 
    return accuracy

def calc_advantage(tst_acc):
    return (tst_acc - .62) / (1-.62)
    
    
    
    
## Saves pre-defined history parameters that keras training returns. It assumes existence of validation data.
def save_history(training_model, history, seed=None, key_idx=None, att=None, valid_val=None):
    # SAVE HISTORY
    ## SAVE HISTORY: LOSS per epoch
    save_file(training_model, history.history['loss'], case=LOSS_CONST, seed=seed, key=key_idx)
    ## SAVE HISTORY: ACCURACY per epoch
    save_file(training_model, history.history['accuracy'], case=ACC_CONST, seed=seed, key=key_idx)
    if valid_val is not None:
        ## SAVE HISTORY: VAL_LOSS per epoch
        save_file(training_model, history.history['val_loss'], case=VAL_LOSS_CONST, seed=seed, key=key_idx)
        ## SAVE HISTORY: VAL_ACCURACY per epoch
        save_file(training_model, history.history['val_accuracy'], case=VAL_ACC_CONST, seed=seed, key=key_idx)

    ## SAVE HISTORY: GRAPH
    save_file(training_model, history, case=TRN_GRPH_CONST, seed=seed, key=key_idx, att=att)
    

    
    
    

# fun little thing for naming saved files
# ('/start_of_file_name', '.file_extension', '/file_folder') # The index comes from the constants above
save_list = [('/mdl', '.h5', "/model"),
             ('/trn_lss', '.npy', "/trn_loss"),
             ('/trn_acc', '.npy', "/trn_accuracy"),
             ('/trn_adv', '.npy', "/trn_advantage"),
             ('/trn_grph', '.png', "/graphs"),
             ('/adv_grph', '.png', ""),
             ('/tst_acc', '.npy', ""),
             ('/val_lss', '.npy', "/val_loss"),
             ('/val_acc', '.npy', "/val_accuracy"),
             ('/val_adv', '.npy', "/val_advantage"),
             ('/tst_adv', '.npy', "")]


def get_file_loc(save_loc, case):
    loc = save_loc + save_list[case][2]
    Path(loc).mkdir(parents=True, exist_ok=True)
    return loc

def get_file_path(save_loc, case, seed=None, key=None):
    tmp_str = get_file_loc(save_loc, case) + save_list[case][0]
    
    if seed is not None:
        tmp_str = tmp_str + "_s{:04d}".format(seed)
    
    if key is not None:
        tmp_str = tmp_str + "_k{:02d}".format(key)
    return tmp_str + "{}".format(save_list[case][1])

def save_file(save_loc, file, case=-1, seed=None, key=None, att=None):
    
    file_path = get_file_path(save_loc, case, seed, key)
    print("++ Saving: ", file_path)
    

    if case in [LOSS_CONST, ACC_CONST, VAL_LOSS_CONST, VAL_ACC_CONST, VAL_ADV_CONST]:
        # This means that file is a numpy array
        np.savetxt(file_path, file)
    
    elif case in [TST_ADV_CONST, TST_ACC_CONST]:
        # This means that file is a dictionary
        np.save(file_path, file)
    
    
    elif case in [TRN_GRPH_CONST]:
        # This means we want to save a graph of a singular key
        # The graph needs the accuracy,  arrays
        # File is history
        plt.plot(file.history['accuracy'])
        plt.plot(file.history['loss'])
        if validation_split_const is not None:
            plt.plot(file.history['val_accuracy'])
            plt.plot(file.history['val_loss'])
        tmp_title = 'Training Graph: ' + 'seed' +'{:02d}'.format(seed)
        if key is not None:
            tmp_title = tmp_title +", key: "+'{:02d}'.format(key)
        if not att == None:
            tmp_title = tmp_title + ', att: ' + '{:01d}'.format(att)
        plt.title(tmp_title)
        plt.ylabel('%')
        plt.xlabel('Epoch')
        if validation_split_const is not None:
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
        if validation_split_const is not None:
            plt.plot(file.history['val_advantage'])
        
        tmp_title = 'Advantage Graph: '
        if seed is not None:
            tmp_title = tmp_title + 'seed' +'{:02d}, '.format(seed)
        if key is not None:
            tmp_title = tmp_title +"key: "+'{:02d}, '.format(key)
        if not att == None:
            tmp_title = tmp_title + ', att: ' + '{:01d}'.format(att)
            
        plt.title(tmp_title)
        plt.ylabel('%')
        plt.xlabel('Epoch')
        if validation_split_const is not None:
            plt.legend(['trn. Adv', 'val. Adv'], loc='upper left')
        else:
            plt.legend(['trn. Adv'], loc='upper left')
        plt.savefig(file_path)
        plt.clf()
    else:
        raise ValueError("save_file was called with a wrong case")
        exit(-1)
        
        
        
# Load attacking traces
# TODO: fix this method up and make it work for ASCAD too.
def display_results(models, accuracies, advantage, training_model, DB_TYPE = TYPE_NTRU):
    
    if DB_TYPE not in [TYPE_ASCAD]:
        adv = { i: (seed, key_idx, (advantage[(seed, key_idx)])) for i, (seed, key_idx) in enumerate(models.keys())}
        print("adv: ", adv)
        data = DataFrame.from_dict(adv, orient='index', columns=['seed', 'key_idx', 'adv'])
        catplot(data=data, x="key_idx", y="adv", row="seed", kind="bar")

        plt.savefig(get_file_path(training_model, ADV_GRPH_CONST))

    else:
        adv = { i: (seed, (advantage[(seed)])) for i, (seed) in enumerate(models.keys())}
        print("adv: ", adv)
        data = DataFrame.from_dict(adv, orient='index', columns=['seed', 'adv'])
        catplot(data=data, y="adv", x="seed", kind="bar")
        plt.savefig(get_file_path(training_model, ADV_GRPH_CONST))
    