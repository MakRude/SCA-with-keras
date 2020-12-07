# SCA with Keras

This is a small framework for implementing, running and testing Machine Learning-based Side Channel Attacks.
As the title says, I rely heavily on Keras in the example, but any Machine Learning model with an identical API should work absolutely fine. 

## Structure

### Load parameters

There are two ways of running the program

#### Running main with arguments:

This is great for automation or if you're looking to run it directly from the terminal.

    $ python main.py <path_to_database> <data_type> <model_type> <save_location> <epochs> <learning_rate> <batch_size>

Example:

    $ python main.py "../GS.pickle" 2 "mlp" "../refactored/testtesttest2" 2 0.00001 100

#### Using hardcoded parameters:
If no arguments are given, then the arguments in the beginning of main will be used. Not that several examples are given.
The same arguments in the examples can be used in the preceding section.

Example:

    $ python main.py 
    

### Load data
There are 6 supported data types: ASCAD, Kyber<sup>1</sup>, NTRU<sup>1</sup>, M4SC, DPA<sup>2</sup>, and Gauss<sup>2</sup>. I relied on a super class through which the main attributes of a Data base are to be met for the program to work.
All classes are saved in the directory <B>f01_data_type</B> as well as the super class, that all database classes inherit.

<B>Note:</B> This means that you could easily impelement a data base of your own and run it if it inherits the super class <B>generic_dl.py</B>.\
<sup>1</sup> As specified in the M4SC repository. \
<sup>2</sup> Gauss was badly implemented and is only there for reference's sake. DPA uses an undocumented format of the DPA contest data and is also only kept in as an example.

### Create Model


### Train model

### Test model

### Save results

## Required library versions:

