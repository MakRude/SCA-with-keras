import training_modules.f01_data_type.generic_dl as gdl
import pickle
import sys
import numpy as np
from keras.utils import to_categorical

class M4sc(gdl.GDL):
    """

    """

    def __load_from_file(self, m4sc_file):
        """
        TODO: add description
        :param m4sc_file:
        :return:
        """

        # TODO: GET THIS DISGUSTING 2-LINER OUT OF HERE AND PROPERLY HANDLE THE UNPICKLING
        # The problem boils down to the unpicked file containing a data type no in the scope of my program that can be imported from m4sc. The type is not relevant to my code but I can't unpickle without loading it. Is there a way around that?
        # TODO: DELETE M4SC REP FROM MY REP
        sys.path.append("./m4sc/")
        import m4sc  # needed in case cw sends faulty data (it's only needed so we can unpickle the data.)
        gdl.check_file_exists(m4sc_file)
        self.__raw_data = []
        try:
            with open(m4sc_file, 'rb') as f:
                while True:
                    try:
                        self.__raw_data.append(pickle.load(f))
                    except EOFError:
                        break
        except:
            print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % m4sc_file)
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
        labels = np.array([i[1] for i in self.__raw_data])
        labels = [[i for i in j] for j in labels]  # list of strings to list of lists
        labels = [np.reshape(i, (24, 2)) for i in labels]  # list of lists to list of matrices (row per coeff in each matrix)
        labels = [[''.join(i) for i in j] for j in labels]  # list of coeff. list

        coeff2num_dict = {
            '11': -1,
            '00': 0,
            '01': 1,
            'ee': 2
        }
        labels = np.array([[coeff2num_dict.get(i) for i in j] for j in labels], dtype=int)
        labels = to_categorical(labels, num_classes=self.num_classes)
        # Normalizing traces of m4sc
        traces = traces - traces.mean()
        traces = traces / traces.max()
        # SET DATA RELATED GLOBALS REQUIRED FOR EXTRACTION
        sample_len = traces.shape[1]  # length of singular trace
        trace_num = traces.shape[0]  # number of traces
        self._set_dims(sample_len, trace_num)
        return traces, labels

    def __init__(self, ascad_file):
        """
        TODO: add description
        :param ascad_file:
        """
        super().__init__(key_ids_num=24, num_classes=4, test_len=2000)
        self.__raw_data = None
        self.__load_from_file(ascad_file)
        self.traces, self.labels = self.__parse_data()
