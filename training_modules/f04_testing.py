import numpy as np

# load model from file
from keras.models import load_model


# This method was taken from the ASCAD code and adapted very heavily

def calc_key_accuracy(data, model, seed=None, key_idx=None):
    """
    TODO: add description
    :param data:
    :param model:
    :param seed:
    :param key_idx:
    :return:
    """

    assert data.loaded_key_idx == key_idx

    x_attack = data.X_testing
    y_attack = data.Y_testing

    if data.key_ids_num > 1:
        tmp_output = "++ calculating accuracy for seed {} and key {}".format(seed, key_idx)
    else:
        tmp_output = "++ calculating accuracy for seed {}".format(seed)
    print(tmp_output)

    # Get the input layer shape
    # input_layer_shape = best_model.get_layer(index=0).input_shape
    # # Adapt the data shape according our model input
    # if len(input_layer_shape) == 2:
    #     # This is a MLP
    #     Reshaped_X_attack = X_attack
    # elif len(input_layer_shape) == 3:
    #     # This is a CNN: expand the dimensions
    #     Reshaped_X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
    # else:
    #     print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
    #     sys.exit(-1)

    predictions = np.argmax(model.predict(x_attack), 1)
    accuracy = sum(np.argmax(y_attack, 1) == predictions)/data.get_test_num()

    return accuracy


def calc_advantage(tst_acc):
    """
    TODO: add description
    :param tst_acc:
    :return:
    """
    return (tst_acc - .62) / (1-.62)


def calc_accuracy(data, models, seeds):
    """
    saves the number of wrongly

    :param data:
    :param models:
    :param seeds: random seeds that were trained (list of ints)
    :return:
    """

    # Assert that this function is only called when several keys exist
    if data.key_ids_num == 1:
        results = dict()

        for seed_x in seeds:
            key_idx = 0
            y_result = np.zeros((data.Y_testing.shape[0], data.key_ids_num))

            model = load_model(models.get((seed_x, key_idx)))

            data.extract_data(key_idx)
            x_attack = data.X_testing
            y_attack = data.Y_testing

            predictions = np.argmax(model.predict(x_attack), 1)
            expected_arr = np.argmax(y_attack, 1)
            y_result[:, 0] = (expected_arr == predictions)

            results[seed_x] = y_result.sum(axis=0)/y_result.shape[0]

        return results, None, None
    
    results = dict()
    total_res = dict()
    hamming_dist = dict()

    for seed_x in seeds:
        y_result = np.zeros((data.Y_testing.shape[0], data.key_ids_num))

        for key_idx in range(data.key_ids_num):
            model = load_model(models.get((seed_x, key_idx)))

            # it's a bit inefficient, because extract data also loads the training data, which is no longer relevant
            data.extract_data(key_idx)
            x_attack = data.X_testing
            y_attack = data.Y_testing

            predictions = np.argmax(model.predict(x_attack), 1)
            expected_arr = np.argmax(y_attack, 1)
            y_result[:, key_idx] = (expected_arr == predictions)

            pass

        total_res[seed_x] = y_result.prod(axis=1)/y_result.shape[0]
        hamming_dist[seed_x] = y_result.shape[1] - y_result.sum(axis=1)
        results[seed_x] = y_result.sum(axis=0)/y_result.shape[0]

        pass
    print("results shape: {}".format(results))
    return results, total_res, hamming_dist
