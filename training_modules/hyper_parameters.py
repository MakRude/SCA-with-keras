# #################################################################################################################
# ################## 1st set of Hyper parameters: #################################################################
# #################################################################################################################
# Visualization and data sampling
tot = 0  # total number of random samples that you'd like to display
f_s = 100  # frequency of data in Hz --- I can't seem to find this value in proj


# ####################################################################################################################
# ################## 2nd set of Hyper parameters: ####################################################################
# ####################################################################################################################


MIN_ACC = 0.95
MAX_ATTEMPTS_PER_KEY = 1
"""
The preceding two constants specify the following:
If the success (Training Accuracy) of our model is lower than MIN_ACC,
then repeat the training with the hopes that new initial weight values would reap better results.
Never run the training of a singular model more than MAX_ATTEMPTS_PER_KEY time 
    - (MAX_ATTEMPTS_PER_KEY = 1) means that no repetitions take place.
"""

drop_out = 0.2
VALIDATION_SPLIT_CONST = 0.1  # (None or a float in }0,1{)
