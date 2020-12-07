import training_modules.f01_data_type.generic_dl as gdl
import h5py
import sys
import numpy as np
from keras.utils import to_categorical


class Ascad(gdl.GDL):
    """
    TODO: add description
    """

    def __load_from_file(self, ascad_file):
        """
        TODO: add description
        :param ascad_file:
        :return:
        """
        gdl.check_file_exists(ascad_file)
        # Open the ASCAD database HDF5 for reading
        try:
            self.__raw_data = h5py.File(ascad_file, "r")
        except:
            print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_file)
            sys.exit(-1)

    def __parse_data(self):
        """
        TODO: add description
        :return:
        """
        assert self.__raw_data is not None
        # Load profiling traces
        x_profiling = np.array(self.__raw_data['Profiling_traces/traces'], dtype=np.int8)
        # Load profiling labels
        y_profiling = np.array(self.__raw_data['Profiling_traces/labels'])
        # Load attacking traces
        x_attack = np.array(self.__raw_data['Attack_traces/traces'], dtype=np.int8)
        # Load attacking labels
        y_attack = np.array(self.__raw_data['Attack_traces/labels'])
        # we don't care about the validation so we're using all the data
        traces = np.concatenate((x_profiling, x_attack), axis=0)
        labels = np.concatenate((y_profiling, y_attack), axis=0)
        labels = to_categorical(labels, num_classes=self.num_classes)
        # SET DATA RELATED GLOBALS REQUIRED FOR EXTRACTION
        sample_len = traces.shape[1]  # length of singular trace
        trace_num = traces.shape[0]  # number of traces
        # set dimension relevant to data extraction for a singular training
        self._set_dims(sample_len, trace_num)
        return traces, labels

    def __init__(self, ascad_file):
        """
        TODO: add description
        :param ascad_file:
        """
        super().__init__(key_ids_num=1, num_classes=256, test_len=2000)
        self.__raw_data = None
        self.__load_from_file(ascad_file)
        self.traces, self.labels = self.__parse_data()


