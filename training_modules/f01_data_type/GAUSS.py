import training_modules.f01_data_type.generic_dl as gdl
import pickle
import sys
import numpy as np
from keras.utils import to_categorical


class Gauss(gdl.GDL):

    def __load_from_file(self, gauss_file):
        gdl.check_file_exists(gauss_file)
        # Open the ASCAD database HDF5 for reading
        self.__raw_data = []
        try:

            with open(gauss_file, 'rb') as f:
                while True:
                    try:
                        self.__raw_data.append(pickle.load(f))
                    except EOFError:
                        break
        except:
            print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % gauss_file)
            sys.exit(-1)

    """
    Labels are prepared by breaking down the first value expelled by the Gaussian sampler
    Said value is seperated into three values, which are learnt individually
    This is a terrible idea. Consider not prepping them and maybe using something like a GAN to train them directly.
    """

    def __parse_data(self):
        """
        TODO: add description
        :return:
        """
        assert self.__raw_data is not None
        traces = np.array([i[4] for i in self.__raw_data])
        labels = [i[2] for i in self.__raw_data]
        # prepare labels
        unique = np.unique(labels)
        self.unique_classes = unique
        class_dic = dict([[unique[i], i] for i in range(len(unique))])
        labels = np.vectorize(class_dic.get)(labels)
        labels = np.array([[int('0x{0:0{1}X}'.format(i, 3)[2], 16),
                            int('0x{0:0{1}X}'.format(i, 3)[3], 16),
                            int('0x{0:0{1}X}'.format(i, 3)[4], 16)]
                           for i in labels])
        # set labels to categorical
        labels = to_categorical(labels, num_classes=self.num_classes)
        # normalize traces to expedite training
        traces = traces - traces.mean()
        traces = traces / traces.max()
        # SET DATA RELATED GLOBALS before returning (POST EXTRACTION)
        sample_len = traces.shape[1]  # length of singular trace
        trace_num = traces.shape[0]  # number of traces
        self._set_dims(sample_len, trace_num)
        return traces, labels

    def __init__(self, gauss_file):
        """
        TODO: add description
        :param gauss_file:
        """
        super().__init__(key_ids_num=3, num_classes=16, test_len=2000)
        self.__raw_data = None
        self.__load_from_file(gauss_file)
        self.traces, self.labels = self.__parse_data()
