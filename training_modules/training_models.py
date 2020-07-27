# Training modules
## CNN
### Model modules
from keras.layers import Input, Conv1D, AveragePooling1D, Flatten, Dense, AveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
### training method modules
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
## CNN2
### Model modules
from keras.layers import BatchNormalization, GaussianNoise, MaxPooling1D, Dropout
## MLP
### Model modules
from keras.optimizers import RMSprop
from keras.models import Sequential

# load hyper_parameters:
from hyper_parameters import drop_out, LEARNING_RATE


### MLP model
def mlp(classes=3, SAMPLE_HIGH=0):

    layer_nb = 4
    node = 5
    input_shape = SAMPLE_HIGH
    model = Sequential()
    # model.add(Dense(node, input_dim=1400, activation='relu'))
    model.add(Dense(node, input_dim=input_shape, activation='relu'))
    for i in range(layer_nb-2):
        model.add(Dense(node, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    optimizer = RMSprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


### MLP model
def mlp2(classes=3, SAMPLE_HIGH=0):

    layer_nb = 4
    node = 5
    input_shape = SAMPLE_HIGH
    model = Sequential()
    # model.add(Dense(node, input_dim=1400, activation='relu'))
    model.add(Dense(node, input_dim=input_shape, activation='relu'))
    for i in range(layer_nb-2):
        model.add(Dense(node, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model



### CNN model - Took ASCAD base
def cnn(classes=3, SAMPLE_HIGH=0):
    # From VGG16 design
#     input_shape = SAMPLE_HIGH
    input_shape = (SAMPLE_HIGH,1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(4, 3, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(8, 3, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(16, 3, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(5, activation='relu', name='fc1')(x)
    x = Dense(5, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn')
    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


### CNN model 2 - ASCAD best cnn w/ RMSprop
def cnn2(classes=3, SAMPLE_HIGH=0):
# From VGG16 design
    input_shape = (SAMPLE_HIGH,1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn')
    optimizer = RMSprop(lr=LEARNING_RATE) # usually 0.00001 for this one
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


### CNN model 2 - Simplified ASCAD
def cnn3(classes=3, SAMPLE_HIGH=0):
    # From VGG16 design
    input_shape = (SAMPLE_HIGH,1)
    img_input = Input(shape=input_shape)
    # Block 0
    x = BatchNormalization()(img_input)
    x = GaussianNoise(0.01)(x)
    # Block 1
    x = Conv1D(8, 3, activation='relu', padding='valid', name='block1_conv1')(x)
    x = MaxPooling1D(2, name='block1_pool')(x)
    x = BatchNormalization(name='block1_bn')(x)
    # Block 2
    x = Conv1D(16, 3, activation='relu', padding='valid', name='block2_conv1')(x)
    x = MaxPooling1D(2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(32, 3, activation='relu', padding='valid', name='block3_conv1')(x)
    x = MaxPooling1D(2, name='block3_pool')(x)
    x = BatchNormalization(name='block3_bn')(x)
    # Block 4
    x = Conv1D(64, 3, activation='relu', padding='valid', name='block4_conv1')(x)
    x = MaxPooling1D(2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(64, 3, activation='relu', padding='valid', name='block5_conv1')(x)
    x = MaxPooling1D(2, name='block5_pool')(x)
    x = BatchNormalization(name='block5_bn')(x)
    # Block 6
    x = Conv1D(256, 3, activation='relu', padding='valid', name='block6_conv1')(x)
    x = MaxPooling1D(2, name='block6_pool')(x)
    x = Dropout(drop_out)(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(50, activation='relu', name='fc1')(x)
    x = Dense(50, activation='relu', name='fc2')(x)
    x = Dropout(drop_out)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn')
    optimizer = Adam(lr=LEARNING_RATE) # this one had 0.001
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


### CNN model 4 - VGG16 based from: https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d
def cnn4(classes=3, SAMPLE_HIGH=0):
    # From VGG16 design
    input_shape = (SAMPLE_HIGH,1)
    img_input = Input(shape=input_shape)
    # Block 0
    x = BatchNormalization()(img_input)
    
    # Block 1
    x = Conv1D(32, 3, activation='relu', padding='valid', name='block1_conv1')(x)
    x = MaxPooling1D(2, name='block1_pool')(x)
    x = Dropout(0.25)(x)
    
    # Block 2
    x = Conv1D(64, 3, activation='relu', padding='valid', name='block2_conv1')(x)
    x = MaxPooling1D(2, name='block2_pool')(x)
    x = Dropout(0.25)(x)

    # Block 3
    x = Conv1D(128, 3, activation='relu', padding='valid', name='block3_conv1')(x)
    x = Dropout(0.4)(x)
    
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='fc1')(x)

    x = Dropout(drop_out)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn')
    optimizer = Adam(lr=LEARNING_RATE) # this one had 0.001
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model




### CNN model 5 - VGG16 based from: https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d
## It now has binary cross_entropy
def cnn5(classes=3, SAMPLE_HIGH=0):
    # From VGG16 design
    input_shape = (SAMPLE_HIGH,1)
    img_input = Input(shape=input_shape)
    # Block 0
    x = BatchNormalization()(img_input)
    
    # Block 1
    x = Conv1D(32, 3, activation='relu', padding='valid', name='block1_conv1')(x)
    x = MaxPooling1D(2, name='block1_pool')(x)
    x = Dropout(0.25)(x)
    
    # Block 2
    x = Conv1D(64, 3, activation='relu', padding='valid', name='block2_conv1')(x)
    x = MaxPooling1D(2, name='block2_pool')(x)
    x = Dropout(0.25)(x)

    # Block 3
    x = Conv1D(128, 3, activation='relu', padding='valid', name='block3_conv1')(x)
    x = Dropout(0.4)(x)
    
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='fc1')(x)

    x = Dropout(drop_out)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn')
    optimizer = Adam(lr=LEARNING_RATE) # this one had 0.001
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model



### MLP model
## It now has binary cross_entropy
def mlp3(classes=3, SAMPLE_HIGH=0):
    layer_nb = 6
    node = 8
    input_shape = SAMPLE_HIGH
    model = Sequential()
    # model.add(Dense(node, input_dim=1400, activation='relu'))
    model.add(Dense(node, input_dim=input_shape, activation='relu'))
    for i in range(layer_nb-2):
        model.add(Dense(node, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    optimizer = RMSprop(lr=LEARNING_RATE)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model




