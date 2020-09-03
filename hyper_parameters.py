##################################################################################################################
################### 1st set of Hyper parameters: #################################################################
##################################################################################################################
# Visualization and data sampling
tot = 0 # total number of random samples that you'd like to display
f_s = 100 # frequency of data in Hz --- I can't seem to find this value in proj


# Training
TST_LEN = 2000 # length of testing set - How many traces would you like to allocate to training?

# my_seeds = [634253, 9134, 57935] # training seeds - list of seed for network to be trained on. Useful for replicating results. -- Replication works a lot better if you train a CPU instead of using a GPU, note however that that sacrifices speed.
my_seeds = [57935]


#####################################################################################################################
################### 2nd set of Hyper parameters: ####################################################################
#####################################################################################################################

batch_size = 100
MIN_ACC = 0.95
drop_out = 0.2
MAX_ATTEMPTS_PER_KEY = 1
validation_split_const = 0.1 # (None or a float in }0,1{)


##################################################################################################################
################### Main Hyper parameters: #################################################################
##################################################################################################################

# my_database <string>: path to data file
# DB_TYPE <int>: 
## sets type of parsing for DB...
### (See available data types in SCA-with-keras/training_modules/misc.py)

# network_type <string>:
## Architecture Type -- ATM you can choose between 'mlp', 'cnn', 'cnn2', and 'cnn3'
### (Look at SCA-with-keras/training_modules/training_models.py for a better overview of the available architectures)

# training_model <string>:
## save folder, in which all the training models, graphs and results are saved. It is created once the program starts.
## You may choose an existing program to resume a training.

# DB_title <string>:
## Name of Architecture to be used for display in graphs, can be chosen at random. (Cosmetic...)

# epochs <int>:
## Can be anywhere between 20 and 200 depending on the architecture and the other Hyperparameters

# LEARNING_RATE <float>:
## anwhere between 1e-10 and 1e-1...
### Your choice depends on your chosen architecture, number of epochs and many other hyper-parameters


epochs = 200
LEARNING_RATE = 0.00001


my_database = "../GS.pickle"
DB_title = "Gaussian Sampler"
DB_TYPE = TYPE_GAUSS 

# CNN training
network_type = "mlp" ## ATM: you can choose between 'mlp', 'cnn', 'cnn2', and 'cnn3'
# save folder
training_model = "../refactored/sca_GS_mlp_lr1e-5_ep50" # save file for results










###### Commented out examples ########
# default parameters values
# my_database = "../chipWhisp01/projects/operand_scanning_32" # Loc on Einstein
# DB_title = "operand_scanning_32" # arbitrary name
# DB_TYPE = TYPE_NTRU

# DB_title = my_database = "../schoolbook32/schoolbook32" # loc on Einstein
# # DB_title = "schoolbook32" ## Optional... It's for the graph
# DB_TYPE = TYPE_NTRU

# my_database = "../PRE_MAY_06/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5" # Loc on Einstein
# DB_title = "ASCAD"
# DB_TYPE = TYPE_ASCAD

# my_database = "../dpa_data"
# DB_title = "DPA contest"
# DB_TYPE = TYPE_DPA

# my_database = "../2020-07-30-115642-118eafdd-ntruprime.pickle"
# DB_title = "M4SC"
# DB_TYPE = TYPE_M4SC

