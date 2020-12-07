# generalist libraries
import sys
import numpy as np
# from random import python_random # TODO: old seed method... Not working
import random
import tensorflow as tf
import os

# load initial arguments
from training_modules.f00_load_parameters import read_parameters_from_file
# Loading data
# # constants for Data types
from training_modules.f01_load_data import TYPE_ASCAD, TYPE_NTRU, TYPE_GAUSS, TYPE_DPA, TYPE_M4SC, TYPE_KYBER
# # data loading meta_function
from training_modules.f01_load_data import load_data
# model related libraries and methods
from keras.models import load_model
from training_modules.f02_deep_learning_models import get_model
# training
from training_modules.f03_train_models import train_model
# testing
from training_modules.f04_handling_saving_stats import save_history, get_file_path, MODEL_CONST, LOSS_CONST, ACC_CONST, \
    ADV_CONST, TRN_GRPH_CONST, ADV_GRPH_CONST, TST_ACC_CONST, VAL_LOSS_CONST, VAL_ACC_CONST, VAL_ADV_CONST, \
    TST_ADV_CONST, CLS_ADV_GRPH_CONST, FIN_ACC_CONST, FIN_ADV_CONST, save_file, display_results
from training_modules.f04_testing import calc_key_accuracy, calc_advantage, calc_accuracy

"""
training seeds -
list of seeds for network to be trained on.
It's useful for replicating results.
Note1: Replication works a lot better if you train a CPU instead of using a GPU.
Note2: Training on a CPU sacrifices speed. This is more noticeable on larger networks.
"""
my_seeds = [57935]


"""
Input explanation:
    my_database <string>: path to data file
    DB_TYPE <int>:
    - sets type of parsing for DB...
    -- See available data types in SCA-with-keras/training_modules/misc.py

    network_type <string>:
    - Architecture Type -- ATM you can choose between 'mlp', 'cnn', 'cnn2', and 'cnn3'
    -- available architectures in file SCA-with-keras/training_modules/f02_deep_learning_models.py

    training_model <string>:
    - save folder, in which all the training models, graphs and results are saved.
    - It is created once the program starts.
    - You may choose an existing save folder when resuming training.

    DB_title <string>:
    - Name of Architecture to be used for display in graphs, can be chosen at random. (Cosmetic...)

    epochs <int>:
    - Can be anywhere between 20 and 200 depending on the architecture and the other Hyperparameters

    LEARNING_RATE <float>:
    - anywhere between 1e-10 and 1e-1...
    -- Your choice depends on your chosen architecture, number of epochs and many other hyper-parameters
"""
if __name__ == "__main__":
    # STEP 1: load parameters
    print("# load parameters")
    if len(sys.argv) != 2:
        # If Program has no input, load pre-defined parameters.

        # # - A few commented out examples
        # default parameters values
        # my_database = "path/to/directory/operand_scanning_32"
        # my_database = "/Users/MakRude/Desktop/general/bachelor/TU_Berlin_SS20/Bachelor Thesis/schoolbook32/" \
        #       + "schoolbook32"
        # data_type = TYPE_NTRU
        # network_type = "cnn"
        # network_type = "mlp"
        # epochs = 200
        # epochs = 10
        # learning_rate = 0.000001
        # learning_rate = 0.0001
        # batch_size = 100
        # training_model = "../refactored/NTRU_testing_refactoring"

        # my_database = "/Users/MakRude/Desktop/general/bachelor/TU_Berlin_SS20/ASCAD.h5"
        # data_type = TYPE_ASCAD
        # # network_type = "cnn2"
        # network_type = "mlp"
        # # epochs = 200
        # epochs = 1
        # learning_rate = 0.0001
        # batch_size = 100
        # training_model = "../refactored/ASCAD_result"
        # # training_model = "../refactored/sca_ascad_cnn2_lr1e-4_ep100_batchSize200"

        # # my_database = "path/to/file/2020-07-30-115642-118eafdd-ntruprime.pickle"
        # my_database = "/Users/MakRude/Desktop/general/bachelor/TU_Berlin_SS20/2020-07-30-115642-118eafdd-ntruprime.pickle"
        # data_type = TYPE_M4SC
        # # network_type = "cnn5"
        # network_type = "mlp"
        # # epochs = 25
        # epochs = 2
        # learning_rate = 0.00001
        # batch_size = 200
        # training_model = "../refactored/m4sc_result"


        # my_database = "path/to/file/2020-09-02-144410-56b4930e-kyber.pickle"
        my_database = "/Users/MakRude/Desktop/general/bachelor/TU_Berlin_SS20/2020-09-02-144410-56b4930e-kyber.pickle"
        data_type = TYPE_KYBER
        network_type = "mlp"
        # epochs = 25
        epochs = 2
        learning_rate = 0.00001
        batch_size = 200
        training_model = "../refactored/kyber_result"

        

        # # my_database = "path/to/gauss/GS.pickle"
        # my_database = "/Users/MakRude/Desktop/general/bachelor/TU_Berlin_SS20/GS50k.pickle"
        # # can be set to: TYPE_ASCAD, TYPE_NTRU, TYPE_GAUSS, TYPE_DPA, TYPE_M4SC
        # data_type = TYPE_GAUSS
        # # check f02_deep_learning_models.py and f02_load_training_model.py to
        # #   figure out what network models are available
        # network_type = "mlp"
        # epochs = 2
        # learning_rate = 0.00001
        # batch_size = 100
        # # TODO: give the save folder a more sensible name:
        # training_model = "../refactored/Gauss_result"
    else:
        # Else: Load parameters from program input

        # get parameters from user input
        # example:
        # $ python main.py "../GS.pickle" 2 "mlp" "../refactored/testtesttest2" 2 0.00001 100

        my_database, data_type, network_type, training_model, epochs, learning_rate, \
            batch_size = read_parameters_from_file(sys.argv[1])
        DB_title = my_database

    # STEP 2: load data
    print("# load data")
    data = load_data(data_type, my_database)
    print("# data successfully loaded")

    # initialize models: dictionary with strings that show paths to model files
    models = dict()
    # initialize histories, accuracies and advantages
    history = dict()
    accuracies = dict()
    advantages = dict()

    # STEP 3: go throw seeds and keys
    print("# go through seeds and keys")
    for seed in my_seeds:
        # increasing reproducibility of results according to https://keras.io/getting_started/faq/
        # set PYTHONHASHSEED before running program to increase reproducibility
        # You can also consider using the CPU, as GPUs are less deterministic. (CUDA="")
        np.random.seed(seed)
        random.seed(seed)
        # tf.random.set_random_seed(seed) # older tf versions use this
        tf.random.set_seed(seed)

        # number of keys is extracted from loaded data
        for key_idx in range(data.key_ids_num):

            
            # STEP 4: load older model if this config. was trained in the past
            # Check if this seed key combination had been trained in the past
            # # we run individual training for each key integer
            if models.get((seed, key_idx)) is not None:
                model = load_model(models.get((seed, key_idx)))
                # STEP XX: extract data for training.
                data.extract_data(key_idx, model=model)

                # # calculate ACCURACY
                accuracies[(seed, key_idx)] = calc_key_accuracy(data, model, seed=seed, key_idx=key_idx)
                # # calculate ADVANTAGE
                advantages[(seed, key_idx)] = calc_advantage(accuracies.get((seed, key_idx)))
                continue

            _file_path = os.path.normpath(get_file_path(training_model, MODEL_CONST, seed, key_idx))
            if os.path.exists(_file_path):
                models[(seed, key_idx)] = get_file_path(training_model, MODEL_CONST, seed, key_idx)
                model = load_model(models.get((seed, key_idx)))
                data.extract_data(key_idx, model=model)
                # # calculate ACCURACY
                accuracies[(seed, key_idx)] = calc_key_accuracy(data, model, seed=seed, key_idx=key_idx)
                # # calculate ADVANTAGE
                advantages[(seed, key_idx)] = calc_advantage(accuracies.get((seed, key_idx)))
                continue

            # STEP 5: initialize model with params selected in arguments and data features.
            model = get_model(network_type, data.num_classes, input_size=data.sample_len, lr=learning_rate)
            data.extract_data(key_idx, model=model)
            # STEP 6: if no older model exists, train the model
            history, att = train_model(data, model, training_model, epochs=epochs, batch_size=batch_size, seed=seed,
                                       key_idx=key_idx)

            # STEP 7.a: save singular training history to file
            save_history(training_model, history, seed, key_idx, att)

            # STEP 7.b: save singular training results in dicts
            # TODO: untested
            # # calculate test accuracy, you'll be saving them all at the end
            accuracies[(seed, key_idx)] = calc_key_accuracy(data, model, seed=seed, key_idx=key_idx)

            # # calculate ADVANTAGE
            advantages[(seed, key_idx)] = calc_advantage(accuracies.get((seed, key_idx)))

            # STEP 7.c: save training model in list of models
            # add file path to list of trained models
            models[(seed, key_idx)] = get_file_path(training_model, MODEL_CONST, seed, key_idx)

            # STEP 8: training for this set-up (seed-data-model combination) is complete.
            pass

        # STEP 9: test results further for all keys combined
        # # this constitutes the attack phase in a Side-Channel Attack
        seed_acc, _, _ = calc_accuracy(data, models, my_seeds)
        save_file(training_model, seed_acc, case=FIN_ACC_CONST)

        # TODO (incomp.): STEP 10: save results to file and display results (also those from STEP 7.b)
        save_file(training_model, accuracies, case=TST_ACC_CONST)
        save_file(training_model, advantages, case=TST_ADV_CONST)

        # # TODO: this only works for one key!!
        # predictions = np.argmax(data_loader.model.predict(X_testing), 1)
        # class_acc = {}
        # class_adv = {}
        #
        # for i in np.unique(Y_testing):
        #     print("in!!")
        #     cls_ind = np.array(np.where(np.array(Y_testing == i))[0])
        #
        #     print("cls_ind.size: ", cls_ind.size)
        #     if cls_ind.size != 0:
        #         cor_guess = len(np.where(np.equal(np.array(Y_testing)[cls_ind], np.array(predictions[cls_ind])))[0])
        #         tot_guess = np.array(Y_testing)[cls_ind].size
        #
        #         #                         print("X_testing: ", X_testing)
        #         print(
        #             "######### Test res for key {:02d} and class {:04d}".format(key_idx, i),
        #             "### {} correct guesses out of a total of {} traces".format(cor_guess, tot_guess),
        #             sep="\n")
        #         class_acc[(seed, key_idx, i)] = cor_guess / tot_guess
        #         class_adv[(seed, key_idx, i)] = calc_advantage(class_acc.get((seed, key_idx, i)))
        #
        # # save advantage results as a graph
        #
        # from seaborn import catplot
        # from pandas import DataFrame
        # import matplotlib.pyplot as plt
        #
        # print("class_adv: ", class_adv)
        #
        # _class_adv = \
        #     {i: (seed, i, (class_adv[(seed, key_idx, i)])) for i, (seed, key_idx, i) in enumerate(class_adv.keys())}
        #
        # data = DataFrame.from_dict(_class_adv, orient='index', columns=['seed', 'cls_ind', 'class_adv'])
        # catplot(data=data, x="cls_ind", y="class_adv", row='seed', kind="bar")
        # plt.savefig(get_file_path(training_model, CLS_ADV_GRPH_CONST))

        # TODO: FOLLOWING COMMENT IS MISSING
        # # print out Confusion Matrix to screen
        #
        # from sklearn.metrics import classification_report, confusion_matrix
        #
        # print('Confusion Matrix')
        # print(confusion_matrix(Y_testing, predictions))
        # print('Classification Report')
        # target_names = ['0', '1', '2', '236', '237']
        #
        # print(classification_report(Y_testing, predictions, target_names=target_names))

        display_results(models, accuracies, advantages, training_model, data_key_num=data.key_ids_num)

        pass

    # TODO: done :)
    pass
