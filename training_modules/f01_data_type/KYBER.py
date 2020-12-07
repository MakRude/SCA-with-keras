import training_modules.f01_data_type.generic_dl as gdl
import pickle
import sys
import numpy as np
from keras.utils import to_categorical


class Kyber(gdl.GDL):

    def __load_from_file(self, kyber_file):
        """
        TODO: add description
        :param kyber_file:
        :return:
        """

        # TODO: GET THIS DISGUSTING 2-LINER OUT OF HERE AND PROPERLY HANDLE THE UNPICKLING
        # TODO: DELETE M4SC REP FROM MY REP
        # These two lines are needed in case cw sends faulty data (it's only needed so we can unpickle the data.)
        sys.path.append("./m4sc/")
        import m4sc
        gdl.check_file_exists(kyber_file)
        # Open the ASCAD database HDF5 for reading
        self.__raw_data = []
        try:

            with open(kyber_file, 'rb') as f:
                while True:
                    try:
                        self.__raw_data.append(pickle.load(f))
                    except EOFError:
                        break
        except:
            print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % kyber_file)
            sys.exit(-1)

    def __parse_data(self):
        """
        TODO: add description
        :return:
        """

        assert self.__raw_data is not None
        # Load traces
        traces = np.array([i[4] for i in self.__raw_data])
        # Load labels
        labels = [i[1] for i in self.__raw_data]
        # process labels
        labels = np.array([bytearray.fromhex(i)[4] for i in labels])
        my_dict = dict([(val, i) for i, val in enumerate(np.unique(labels))])
        labels = np.vectorize(my_dict.get)(labels)
        labels = to_categorical(labels, num_classes=self.num_classes)
        print("converting labels to ascending values as follows: \n", my_dict)
        traces = traces - traces.mean()
        traces = traces / traces.max()
        # SET DATA RELATED GLOBALS REQUIRED FOR EXTRACTION
        sample_len = traces.shape[1]  # length of singular trace
        trace_num = traces.shape[0]  # number of traces
        # set dimension relevant to data extraction for a singular training
        self._set_dims(sample_len, trace_num)
        return traces, labels

    def __parse_data_no_zeros(self):
        """
        TODO: add description
        :return:
        """

        assert self.__raw_data is not None
        # Load traces
        traces = [i[4] for i in self.__raw_data]
        # Load labels
        labels = [i[1] for i in self.__raw_data]
        # remove zeroes labels from data
        labels = np.array([bytearray.fromhex(i)[4] for i in labels])
        traces = np.array([val for ind, val in enumerate(traces) if labels[ind] != 0])
        labels = labels[labels != 0]
        my_dict = dict([(val, i) for i, val in enumerate(np.unique(labels))])
        labels = np.vectorize(my_dict.get)(labels)
        labels = to_categorical(labels, num_classes=self.num_classes)
        print("converting labels to ascending values as follows: \n", my_dict)
        # Normalizing traces of Kyber
        traces = traces - traces.mean()
        traces = traces / traces.max()
        # SET DATA RELATED GLOBALS REQUIRED FOR EXTRACTION
        sample_len = traces.shape[1]  # length of singular trace
        trace_num = traces.shape[0]  # number of traces
        self._set_dims(sample_len, trace_num)
        return traces, labels

    def __init__(self, kyber_file):
        """
        TODO: add description
        :param kyber_file:
        """

        super().__init__(key_ids_num=1, num_classes=6, test_len=2000)
        self.__raw_data = None
        self.__load_from_file(kyber_file)
        self.traces, self.labels = self.__parse_data()
