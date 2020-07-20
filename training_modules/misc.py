import os



# Type constants defined as follows:
TYPE_ASCAD = 0
TYPE_NTRU = 1
TYPE_GAUSS = 2
TYPE_DPA = 3
TYPE_M4SC = 4
    
    
    
# file handling constants defined as follows
# this is relevant for the saving functionality when saving and loading arrays, models, and graphs
MODEL_CONST = 0
LOSS_CONST = 1
ACC_CONST = 2
ADV_CONST = 3
TRN_GRPH_CONST = 4
ADV_GRPH_CONST = 5
TST_ACC_CONST = 6
VAL_LOSS_CONST = 7
VAL_ACC_CONST = 8
VAL_ADV_CONST = 9
TST_ADV_CONST = 10

# ASCAD: snippet from ASCAD code that's used throughout

def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

