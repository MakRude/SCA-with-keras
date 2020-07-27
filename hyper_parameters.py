##################################################################################################################
################### 1st set of Hyper parameters: #################################################################
##################################################################################################################
# Visualization and data sampling
tot = 0 # total number of random samples that you'd like to display
f_s = 100 # frequency of data in Hz --- I can't seem to find this value in proj


# Training
TST_LEN = 50 # length of testing set - How many traces would you like to allocate to training?
# my_seeds = [634253, 9134, 57935] # training seeds - list of seed for network to be trained on. Useful for replicating similar results.
my_seeds = [57935]


#####################################################################################################################
################### 2nd set of Hyper parameters: ####################################################################
#####################################################################################################################

epochs = 200
batch_size = 100
MIN_ACC = 0.95
drop_out = 0.2
MAX_ATTEMPTS_PER_KEY = 1
LEARNING_RATE = 0.01
validation_split_const = 0.1 # (None or a float in }0,1{)
