import training_modules.f01_data_type.generic_dl as gdl
import sys
import chipwhisperer as cw
import numpy as np
from numpy import empty
from keras.utils import to_categorical


class Ntru(gdl.GDL):
    """
    TODO: add description
    """

    def __load_from_file(self, ntru_file):
        """
        TODO: add description


        :param ntru_file:
        :return:
        """
        
        gdl.check_file_exists(ntru_file+".cwp")

        # Open the ASCAD database HDF5 for reading
        # try:
        self.__raw_data = cw.open_project(ntru_file)
        self.key_num = self.__raw_data.keys[0].size # 64
        self.key_ids_num = int(self.key_num/2)      # 32

        # except:
            # print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ntru_file)
            # sys.exit(-1)

    def __parse_data(self):
        """
        TODO: add description
        :return:
        """
        assert self.__raw_data is not None
        # Load traces
        # SET DATA RELATED GLOBALS REQUIRED FOR EXTRACTION
        sample_len = self.__raw_data.trace_manager().num_points()  # length of singular trace
        trace_num = len(self.__raw_data.traces)  # number of traces

        # set dimension relevant to data extraction for a singular training
        self._set_dims(sample_len, trace_num)

        traces = empty(shape=(self.sample_num, self.sample_len))
        for i in range(self.sample_num):
            traces[i] = self.__raw_data.waves[i][self._sample_slice]
        # Load labels
        labels = empty(shape=(self.sample_num, self.key_num))
        for i in range(self.sample_num):
            # reproduce values
            labels[i] = self.__raw_data.keys[i]
            
        # process labels
        labels = np.array((labels.T[1::2].T * (2**8)) + labels.T[::2].T, dtype=np.int16)
        unique = np.unique(labels)
        class_dic = dict([[unique[i], i] for i in range(len(unique))])
        labels = np.vectorize(class_dic.get)(labels)
        labels = to_categorical(labels, num_classes=self.num_classes)
        
        return traces, labels

    def __init__(self, file):
        """
        TODO: add description
        :param ascad_file:
        """
        super().__init__(key_ids_num=1, num_classes=3, test_len=2000)
        self.__raw_data = None
        self.__load_from_file(file)
        self.traces, self.labels = self.__parse_data()
