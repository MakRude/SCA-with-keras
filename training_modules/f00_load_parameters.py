

def read_parameters_from_file(param_filename):
    """
    TODO: add description
    :param param_filename:
    :return:
    """
    # read parameters for the train_model and load_traces functions
    # TODO: sanity checks on parameters
    param_file = open(param_filename,"r")

    # TODO: replace eval() by ast.linear_eval()
    my_parameters = eval(param_file.read())

    my_database = my_parameters["my_database"]
    DB_TYPE = my_parameters["DB_TYPE"]
    network_type = my_parameters["network_type"]
    training_model = my_parameters["training_model"]
    epochs = my_parameters["epochs"]
    LEARNING_RATE = my_parameters["LEARNING_RATE"]
    batch_size = my_parameters["batch_size"]

    return my_database, DB_TYPE, network_type, training_model, epochs, LEARNING_RATE, batch_size
