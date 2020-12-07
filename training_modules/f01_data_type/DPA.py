import training_modules.f01_data_type.generic_dl as gdl
import sys
import numpy as np
from keras.utils import to_categorical


class Dpa(gdl.GDL):
    """

    """

    def __load_from_file(self, dpa_file):
        """
        TODO: add description
        :param dpa_file:
        :return:
        """
        gdl.check_file_exists(dpa_file)
        # Open the ASCAD database HDF5 for reading
        try:
            x = np.load(dpa_file + "/traces.npy")
            y = np.load(dpa_file + "/labels_pt.npy")
            self.__raw_data = (x, y)
        except:
            print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % dpa_file)
            sys.exit(-1)

    def __parse_data(self):
        """
        TODO: add description
        :return:
        """
        assert self.__raw_data is not None
        traces, labels = self.__raw_data

        # set labels to categorical
        labels = to_categorical(labels, num_classes=self.num_classes)
        # SET DATA RELATED GLOBALS REQUIRED FOR EXTRACTION
        sample_len = traces.shape[1]  # length of singular trace
        trace_num = traces.shape[0]  # number of traces
        # set dimension relevant to data extraction for a singular training
        self._set_dims(sample_len, trace_num)
        return traces, labels

    def __init__(self, dpa_file):
        """
        TODO: add description
        :param dpa_file:
        """
        super().__init__(key_ids_num=1, num_classes=2, test_len=2000)
        self.__raw_data = None
        self.__load_from_file(dpa_file)
        self.traces, self.labels = self.__parse_data()


