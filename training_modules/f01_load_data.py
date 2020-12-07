# general modules
import sys
# # training models
# from training_modules.training_models import mlp, mlp2, mlp3, cnn, cnn2, cnn3, cnn4, cnn5, cnn2_2

from .f01_data_type.ASCAD import Ascad
from .f01_data_type.DPA import Dpa
from .f01_data_type.GAUSS import Gauss
from .f01_data_type.KYBER import Kyber
from .f01_data_type.M4SC import M4sc
from .f01_data_type.NTRU import Ntru


# Type constants defined as follows:
TYPE_ASCAD = "ascad"
TYPE_NTRU = "ntru"
TYPE_GAUSS = "gauss"
TYPE_DPA = "dpa"
TYPE_M4SC = "m4sc"
TYPE_KYBER = "kyber"


def load_data(db_type, my_database):
    """
    TODO: add description
    :param db_type:
    :param my_database:
    :return:
    """
    if db_type in [TYPE_ASCAD]:
        data = Ascad(my_database)
        data.dataType = TYPE_ASCAD

    elif db_type in [TYPE_NTRU]:
        data = Ntru(my_database)
        data.dataType = TYPE_NTRU

    elif db_type in [TYPE_GAUSS]:
        data = Gauss(my_database)
        data.dataType = TYPE_GAUSS

    elif db_type in [TYPE_DPA]:
        data = Dpa(my_database)
        data.dataType = TYPE_DPA

    elif db_type in [TYPE_M4SC]:
        data = M4sc(my_database)
        data.dataType = TYPE_M4SC

    elif db_type in [TYPE_KYBER]:
        data = Kyber(my_database)
        data.dataType = TYPE_KYBER

    else:
        print("This shouldn't happen. DB_TYPE entered is not supported.")
        sys.exit(-1)

    return data
