import sys
import os


# A snippet of code from ASCAD code that's useful for the data loaders.
def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


class GDL:
    """
    a generic data loading super-class, with which data can be loaded from different databases.
    """

    def _set_dims(self, sample_len, trace_num):
        """
        The methods sets data dimensions relevant to data_parsing
        Note:
        Method is a code snippet taken from ASCAD_train_model. Altered by Mahmoud Gharra
        TODO: add description
        :param sample_len:
        :param trace_num:
        :return:
        """

        assert self.__test_len is not None

        self.__sample_high = sample_len  # length of singular trace
        self.__sample_low = 0
        self.sample_len = sample_len

        self._sample_slice = slice(self.__sample_low, self.__sample_high)
        self.sample_num = trace_num  # number of traces

        training_num = self.sample_num - self.__test_len

        self.__training_slice = slice(0, training_num)

        self.__test_num = self.sample_num - training_num
        self.__test_slice = slice(training_num, self.__test_num + training_num)

        assert self.__test_num + training_num <= self.sample_num
        assert training_num > 3 * self.__test_num

    def extract_data(self, key_idx=None, model=None):
        """
        This method sets the profiling and testing traces for a singular training
        Assumes that traces X and labels y are set.

        TODO: add description
        :param key_idx:
        :return:
        """

        assert self.traces is not None
        assert self.labels is not None

        assert key_idx is not None

        self.loaded_key_idx = key_idx

        # Load profiling traces
        self.X_profiling = self.traces[self.__training_slice]
        
        # Get the input layer shape
        # input_layer_shape = model.get_layer(index=0).input_shape
        if model is not None:
            # TODO: sometimes the traces need to be reshaped. This snippet is taking from NIST-FR
            input_layer_shape = model.get_layer(index=0).input_shape
            if input_layer_shape[1] != len(self.X_profiling[0]):
                print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(self.X_profiling[0])))
                sys.exit(-1)

            # Adapt the data shape according our model input
            if len(input_layer_shape) == 2:
                # This is a MLP
                Reshaped_X_profiling = self.X_profiling
            elif len(input_layer_shape) == 3:
                # This is a CNN: expand the dimensions
                Reshaped_X_profiling = self.X_profiling.reshape((self.X_profiling.shape[0], self.X_profiling.shape[1], 1))
            else:
                print("Error: model input shape length %d is not supported ..." % len(input_layer_shape))
                sys.exit(-1)

            self.X_profiling = Reshaped_X_profiling

        # Load testing traces
        self.X_testing = self.traces[self.__test_slice, :]

        # load labels
        if self.key_ids_num > 1:
            self.loaded_key_idx = key_idx
            # Load profiling labels
            self.Y_profiling = self.labels[self.__training_slice, key_idx]
            # Load testing labels
            self.Y_testing = self.labels[self.__test_slice, key_idx]
        elif self.key_ids_num == 1:
            self.loaded_key_idx = key_idx
            # Load profiling labels
            self.Y_profiling = self.labels[self.__training_slice]
            # Load testing labels
            self.Y_testing = self.labels[self.__test_slice]
        else:
            print("Error: This shouldn't happen. number of key_ids is not supported...")
            sys.exit(-1)

        

    def __init__(self, key_ids_num=0, num_classes=0, test_len=0):
        """
        Initializes a generic data loader that can be used as a super-class to handle different databases

        parameter values
        - key_ids_num: number of label types to be predicted from traces
        - num_classes: number of classes used by dataset (number of discrete label values)
        - test_len: test traces size
        TODO: add description
        :param key_ids_num:
        :param num_classes:
        :param test_len:
        """

        self.key_ids_num = key_ids_num
        self.num_classes = num_classes
        self.__test_len = test_len

        self.traces = None
        self.labels = None

        self.X_profiling = None
        self.X_testing = None

        self.Y_profiling = None
        self.Y_testing = None

        # shows currently loaded attack traces
        self.loaded_key_idx = None
        
        self.__test_num = None

    def get_test_num(self):
        """
        Returns numbers of test traces
        """
        assert self.__test_num is not None

        return self.__test_num
