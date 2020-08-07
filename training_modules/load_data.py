# general modules
import sys
# Load data modules
from numpy import empty, zeros, uint16
from numpy.random import RandomState
import numpy as np
## load cw data
import chipwhisperer as cw
## load ASCAD data
import h5py
## load m4sc data
import pickle

# consts
from hyper_parameters import TST_LEN
from training_modules.misc import TYPE_ASCAD, TYPE_NTRU, TYPE_GAUSS, TYPE_DPA, TYPE_M4SC
from training_modules.misc import check_file_exists
# training models
from training_modules.training_models import mlp, mlp2, mlp3, cnn, cnn2, cnn3, cnn4, cnn5, cnn2_2


class DATA_LOADER:     
    # code snippet taken from ASCAD_train_model. Altered by Mahmoud Gharra (concatinated profiling and attack data to fit our code)

    def __set_dims(self, sample_len, trace_num):
        
        self.SAMPLE_HIGH = sample_len # length of singular trace
        
        self.sample_low = 0
        self.sample_slice = slice(self.sample_low, self.SAMPLE_HIGH)
        self.SAMPLE_NUM = trace_num # number of traces

        training_num = self.SAMPLE_NUM - TST_LEN
        
        self.TRAINING_SLICE = slice(0, training_num)
        
        self.TEST_NUM = self.SAMPLE_NUM - training_num
        self.TEST_SLICE = slice(training_num, self.TEST_NUM + training_num)
        
        assert self.TEST_NUM + training_num <= self.SAMPLE_NUM
        assert training_num > 3*self.TEST_NUM
        
        
        
        
    #### ASCAD helper to load profiling and attack data (traces and labels)
    # Loads the profiling and attack datasets from the ASCAD
    # database
    def __load_ascad(self, ascad_database_file):
        check_file_exists(ascad_database_file)
        # Open the ASCAD database HDF5 for reading
        try:
            in_file  = h5py.File(ascad_database_file, "r")
        except:
            print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
            sys.exit(-1)
        # Load profiling traces
        X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
        # Load profiling labels
        Y_profiling = np.array(in_file['Profiling_traces/labels'])
        # Load attacking traces
        X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
        # Load attacking labels
        Y_attack = np.array(in_file['Attack_traces/labels'])
        
        # we don't care about the validation so we're using all the data
        X = np.concatenate((X_profiling, X_attack), axis=0)
        Y = np.concatenate((Y_profiling, Y_attack), axis=0)
        
        # SET DATA RELATED GLOBALS REQUIRED FOR EXTRACTION
        sample_len = X.shape[1] # length of singular trace
        trace_num = X.shape[0] # number of traces
        
        self.KEY_LENGTH = 1
        self.__set_dims(sample_len, trace_num)
        
        return X, Y
    
    
    # ASCAD: Adapted by Mahmoud Gharra to fit the NTRU Prime input
    
    # returns traces and labels
    def __load_database_cw(self, my_database):
        # load traces
        print("++ Loading projects")
        project = cw.open_project(my_database)

        # Organize trace data for MLP
        print("++ Organizing traces")
        self.KEY_LENGTH = TEXT_LENGTH = project.keys[0].size

        sample_len = project.trace_manager().num_points() # length of singular trace
        trace_num = len(project.traces) # number of traces

        self.__set_dims(sample_len, trace_num)
        
        # organize traces in X matrix
        X = empty(shape=(self.SAMPLE_NUM, self.SAMPLE_HIGH - self.sample_low))
        for i in range(self.SAMPLE_NUM):
            X[i] = project.waves[i][self.sample_slice]

        # organize the operands in Y matrix
        y = empty(shape=(self.SAMPLE_NUM, self.KEY_LENGTH))
        for i in range(self.SAMPLE_NUM):
            # reproduce values
            y[i] = project.keys[i]
            
        # this is in case of little endian
        y = np.array((y.T[1::2].T*(2**8)) + y.T[::2].T, dtype=np.int16)
        self.KEY_LENGTH = int(self.KEY_LENGTH/2)

        # transform generated key numbers to labels
        unique = np.unique(y)
        class_dic = dict([[unique[i], i] for i in range(len(unique))])
        y_labelled = np.vectorize(class_dic.get)(y)
        return X, y_labelled

    
    
    

    # ASCAD: Adapted by Mahmoud Gharra to fit the Gaussian Sampler input


    # returns traces and labels

    def __load_database_gauss(self, my_database):
        # load traces
        print("++ Loading projects")
        project = cw.open_project(my_database)

        # Organize trace data for MLP
        print("++ Organizing traces")
        self.KEY_LENGTH = len(project.keys[0]) * 8
#         print("KEY_LENGTH: {}".format(self.KEY_LENGTH))
        print("project.keys[0][0]: {}".format(project.keys[0][0]))
        print("np.asarray(bytearray(project.keys[0]))", np.asarray(bytearray(project.keys[0])))

        # SET DATA RELATED GLOBALS REQUIRED FOR EXTRACTION
        sample_low = 0
        self.SAMPLE_HIGH = project.trace_manager().num_points() # length of singular trace
        sample_slice = slice(sample_low, self.SAMPLE_HIGH)
        sample_num = len(project.traces) # number of traces
    #     print("sample num: ", sample_num)

        # organize traces in X and Y matrices
        
        # count is used to count traces with wrong label length (some labels displayed a length of 31 instead of 32. This is a quick work-around)
        count = 0

        X = empty(shape=(sample_num, self.SAMPLE_HIGH - sample_low))
        y = empty(shape=(sample_num, KEY_LENGTH))

        for i in range(sample_num):
            if (len(np.asarray(bytearray(project.keys[i])))) is not int(KEY_LENGTH/8):
                count += 1
                print("Number of problematic traces raised to: {}".format(count))
                continue
            # we subtract count because we're trying to fill in for the wrong traces
            y[i-count] = np.asarray([int(char) for char in ''.join(['{0:08b}'.format(ff) for ff in np.asarray(bytearray(project.keys[0]))])])
            X[i-count] = project.waves[i][sample_slice]
            
        # remove last {count} rows (They're empty - some of the data was corrupted)
        if i is not 0:
            y = y[0:-count]
            X = X[0:-count]


        # SET DATA RELATED GLOBALS before returning (POST EXTRACTION)
        sample_len = project.trace_manager().num_points() # length of singular trace
        trace_num = len(project.traces) - count # number of traces
                          
        self.__set_dims(sample_len, trace_num)
                          
        return X, y



    # ASCAD: Adapted by Mahmoud Gharra to fit the Gaussian Sampler input
    
    
    # returns traces and labels

    def __load_database_dpa_contest(self, my_database):
        # load traces
        print("++ Loading dpa contest traces")
        X = np.load(my_database + "/traces.npy")
        
        y_init = np.load(my_database + "/labels_pt.npy")
        y = np.array([[int(x) for x in '{:08b}'.format(int(input_init))] for input_init in y_init[:,0]])
        
        # SET DATA RELATED GLOBALS REQUIRED FOR EXTRACTION
        self.KEY_LENGTH = 4
        
                          
        sample_len = X.shape[1] # length of singular trace
        trace_num = X.shape[0] # number of traces
       
        self.__set_dims(sample_len, trace_num)
        
                          
        return X, y



    # these functions were used in the loading of m4sc data
    def __bit8(self, i):
        return [(0 if (0 == (i & (1 << j))) else 1) for j in range(8)]

    def __concatBits(self, a, b):
        return 2*a + b

    def __chunk4(self, i):
        temp = self.__bit8(i)
        return [self.__concatBits(temp[2*i], temp[2*i + 1]) for i in range(4)]

    def __to_16bits(self, num):
        return np.array([0 if (num & (2**i) is 0) else 1 for i in range(16)])

#     def bit16(i):
#     #     return [(0 if (0 == (i & (1 << j))) else 1) for j in range(15, -1, -1)]
#         return [(0 if (0 == (i & (1 << j))) else 1) for j in range(16)]

#     def bit368(i):
#         return [(0 if (0 == (i & (1 << j))) else 1) for j in range(368)]


    # ASCAD: Adapted by Mahmoud Gharra to fit m4sc input

    # returns traces and labels

    def __load_database_m4sc(self, my_database):
        # TODO: GET THIS DISGUSTING 2-LINER OUT OF HERE AND PROPERLY HANDLE THE UNPICKLING
        # TODO: DELETE M4SC REP FROM MY REP
        sys.path.append("./m4sc/")
        import m4sc # needed in case cw sends faulty data (it's only needed so we can unpickle the data.)

        # load traces
        print("++ Loading schoolbook on m4sc data")
        
        data = []
        data_file = my_database
        with open(data_file, 'rb') as f:
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break

        # parse traces - convert int array to bit array
        print("++ Parse m4sc data")
        X = np.array([i[4] for i in data])
#         y_init = np.array([int((int(i[0][:2], 16))) for i in data])
#         y = np.array([self.__bit8(i) for i in y_init], dtype=np.uint8)
        
        y_init = np.array([i[1] for i in data])
        
        y_tmp = [[i for i in j] for j in y_init] # list of strings to list of lists
        y_tmp = [np.reshape(i,(24, 2)) for i in y_tmp] # list of lists to list of matrices (row per coeff in each matrix)
        y_tmp = [[''.join(i) for i in j] for j in y_tmp] # list of coeff. list

        coeff2num_dict = {
            '11':-1,
            '00':0,
            '01':1,
            'ee':2
        }
#         print("y_tmp: ", y_tmp)
        y = np.array([[coeff2num_dict.get(i) for i in j] for j in y_tmp], dtype=int)
#         y = np.array([self.__to_16bits(i) for i in y_init], dtype=np.uint8)
#         y = np.array([int((int(i[0][:2], 16))) for i in data])
        print("++ Finished parsing m4sc data")
        print("++ Normalizing traces of m4sc")
        X = X - X.mean()
        X = X / X.max()


        # Organize trace data for network
        print("++ Organizing traces")
    #     KEY_LENGTH = TEXT_LENGTH = 1
        self.KEY_LENGTH = 24
#         print("KEY_LENGTH: {}".format(self.KEY_LENGTH))

        # SET DATA RELATED GLOBALS REQUIRED FOR EXTRACTION
        sample_len = X.shape[1] # length of singular trace
        trace_num = X.shape[0] # number of traces
        print("Traces shape: {}".format(X.shape))
        self.__set_dims(sample_len, trace_num)
        
                          
        return X, y    
    
    
    
    
    def __init__(self, DB_TYPE, my_database, meta=""):
        self.DB_TYPE = DB_TYPE
        print("+ Commense loading data")
        if DB_TYPE in [TYPE_ASCAD]:
            self.X, self.Y = self.__load_ascad(my_database)
            self.num_classes = 256
            self.TO_CAT = True
        elif DB_TYPE in [TYPE_NTRU]:
            self.X, self.Y = self.__load_database_cw(my_database)
            self.num_classes = 3
            self.TO_CAT = True
        elif DB_TYPE in [TYPE_GAUSS]:
            self.num_classes = 2
            self.X, self.Y = self.__load_database_gauss(my_database)
            self.TO_CAT = True
        elif DB_TYPE in [TYPE_DPA]:
            self.num_classes = 2
            self.X, self.Y = self.__load_database_dpa_contest(my_database)
            self.TO_CAT = True
        elif DB_TYPE in [TYPE_M4SC]:
            self.num_classes = 4
            self.X, self.Y = self.__load_database_m4sc(my_database)
            self.TO_CAT = True
        else:
            print("This shouldn't happen. DB_TYPE entered is not supported.")
            sys.exit(-1)
            
    def extract_data(self, network_type, key_idx=None):
        #get network type
        if(network_type=="cnn"):
            best_model = cnn(self.num_classes, self.SAMPLE_HIGH)
        elif(network_type=="cnn2"):
            best_model = cnn2(self.num_classes, self.SAMPLE_HIGH)
        elif(network_type=="cnn2_2"):
            best_model = cnn2_2(self.num_classes, self.SAMPLE_HIGH)
        elif(network_type=="cnn3"):
            best_model = cnn3(self.num_classes, self.SAMPLE_HIGH)
        elif(network_type=="cnn4"):
            best_model = cnn4(self.num_classes, self.SAMPLE_HIGH)
        elif(network_type=="cnn5"):
            best_model = cnn5(self.num_classes, self.SAMPLE_HIGH)
        elif(network_type=="mlp"):
            best_model = mlp(self.num_classes, self.SAMPLE_HIGH)
        elif(network_type=="mlp2"):
            best_model = mlp2(self.num_classes, self.SAMPLE_HIGH)
        elif(network_type=="mlp3"):
            best_model = mlp3(self.num_classes, self.SAMPLE_HIGH)
        else: #display an error and abort
            print("Error: no topology found for network '%s' ..." % network_type)
            sys.exit(-1);

        self.model = best_model
        # Load profiling traces
        self.X_profiling = self.X[self.TRAINING_SLICE]
        # Load testing traces
        self.X_testing = self.X[self.TEST_SLICE]
        ## make sure input data has correct shape (this part was adapted from ASCAD code as we need to do the exact same thing here)
        input_layer_shape = best_model.get_layer(index=0).input_shape
#         # Adapt the data shape according our model input
#         if len(input_layer_shape) == 2:
#             # This is a MLP
#             self.X_testing = self.X[self.TEST_SLICE, :]
#         elif len(input_layer_shape) == 3:
#             # This is a CNN: reshape the data
#             self.X_testing = self.X[self.TEST_SLICE, :]
#             self.X_testing = self.X_testing.reshape((self.X_testing.shape[0], self.X_testing.shape[1], 1))
#         else:
#             print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
#             sys.exit(-1)

        # Adapt the data shape according our model input
        self.X_testing = self.X[self.TEST_SLICE, :]

        # ASCAD has only one key so it's labels are referenced differently
        if self.DB_TYPE in [TYPE_NTRU, TYPE_GAUSS, TYPE_DPA, TYPE_M4SC]:
            # Load profiling labels
            self.Y_profiling = self.Y[self.TRAINING_SLICE, key_idx]
            # Load testing labels
            self.Y_testing = self.Y[self.TEST_SLICE, key_idx]
        elif self.DB_TYPE in [TYPE_ASCAD]:
            # Load profiling labels
            self.Y_profiling = self.Y[self.TRAINING_SLICE]
            # Load testing labels
            self.Y_testing = self.Y[self.TEST_SLICE]
        else:
            print("Error: This shouldn't happen. DB_TYPE is not supported...")
            sys.exit(-1)

       
           