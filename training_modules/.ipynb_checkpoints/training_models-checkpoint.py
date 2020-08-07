{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training modules\n",
    "## CNN\n",
    "### Model modules\n",
    "from keras.layers import Input, Conv1D, AveragePooling1D, Flatten, Dense, AveragePooling1D\n",
    "from keras.models import Model\n",
    "from keras.optimizers import Adam\n",
    "### training method modules\n",
    "from keras.callbacks import ModelCheckpoint, EarlyStopping\n",
    "from keras.utils import to_categorical\n",
    "from keras.models import load_model\n",
    "## CNN2\n",
    "### Model modules\n",
    "from keras.layers import BatchNormalization, GaussianNoise, MaxPooling1D, Dropout\n",
    "## MLP\n",
    "### Model modules\n",
    "from keras.optimizers import RMSprop\n",
    "from keras.models import Sequential\n",
    "\n",
    "\n",
    "\n",
    "### MLP model\n",
    "def mlp(classes=3):\n",
    "\n",
    "    layer_nb = 4\n",
    "    node = 5\n",
    "    input_shape = SAMPLE_HIGH\n",
    "    model = Sequential()\n",
    "    # model.add(Dense(node, input_dim=1400, activation='relu'))\n",
    "    model.add(Dense(node, input_dim=input_shape, activation='relu'))\n",
    "    for i in range(layer_nb-2):\n",
    "        model.add(Dense(node, activation='relu'))\n",
    "    model.add(Dense(classes, activation='softmax'))\n",
    "    optimizer = RMSprop(lr=0.01)\n",
    "    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])\n",
    "    return model\n",
    "\n",
    "\n",
    "### CNN model - Took ASCAD base\n",
    "def cnn(classes=3):\n",
    "    # From VGG16 design\n",
    "    input_shape = (SAMPLE_HIGH,1)\n",
    "    img_input = Input(shape=input_shape)\n",
    "    # Block 1\n",
    "    x = Conv1D(4, 3, activation='relu', padding='same', name='block1_conv1')(img_input)\n",
    "    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)\n",
    "    # Block 2\n",
    "    x = Conv1D(8, 3, activation='relu', padding='same', name='block2_conv1')(x)\n",
    "    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)\n",
    "    # Block 3\n",
    "    x = Conv1D(16, 3, activation='relu', padding='same', name='block3_conv1')(x)\n",
    "    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)\n",
    "    x = Flatten(name='flatten')(x)\n",
    "    x = Dense(5, activation='relu', name='fc1')(x)\n",
    "    x = Dense(5, activation='relu', name='fc2')(x)\n",
    "    x = Dense(classes, activation='softmax', name='predictions')(x)\n",
    "\n",
    "    inputs = img_input\n",
    "    # Create model.\n",
    "    model = Model(inputs, x, name='cnn')\n",
    "    optimizer = Adam(lr=LEARNING_RATE)\n",
    "    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])\n",
    "    return model\n",
    "\n",
    "\n",
    "### CNN model 2 - ASCAD best cnn w/ adam\n",
    "def cnn2(classes=3):\n",
    "# From VGG16 design\n",
    "    input_shape = (SAMPLE_HIGH,1)\n",
    "    img_input = Input(shape=input_shape)\n",
    "    # Block 1\n",
    "    x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)\n",
    "    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)\n",
    "    # Block 2\n",
    "    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)\n",
    "    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)\n",
    "    # Block 3\n",
    "    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)\n",
    "    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)\n",
    "    # Block 4\n",
    "    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)\n",
    "    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)\n",
    "    # Block 5\n",
    "    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)\n",
    "    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)\n",
    "    # Classification block\n",
    "    x = Flatten(name='flatten')(x)\n",
    "    x = Dense(4096, activation='relu', name='fc1')(x)\n",
    "    x = Dense(4096, activation='relu', name='fc2')(x)\n",
    "    x = Dense(classes, activation='softmax', name='predictions')(x)\n",
    "\n",
    "    inputs = img_input\n",
    "    # Create model.\n",
    "    model = Model(inputs, x, name='cnn')\n",
    "    optimizer = RMSprop(lr=LEARNING_RATE) # usually 0.00001 for this one\n",
    "    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])\n",
    "    return model\n",
    "\n",
    "\n",
    "### CNN model 2 - Simplified ASCAD\n",
    "def cnn3(classes=3):\n",
    "    # From VGG16 design\n",
    "    input_shape = (SAMPLE_HIGH,1)\n",
    "    img_input = Input(shape=input_shape)\n",
    "    # Block 0\n",
    "    x = BatchNormalization()(img_input)\n",
    "    x = GaussianNoise(0.01)(x)\n",
    "    # Block 1\n",
    "    x = Conv1D(8, 3, activation='relu', padding='valid', name='block1_conv1')(x)\n",
    "    x = MaxPooling1D(2, name='block1_pool')(x)\n",
    "    x = BatchNormalization(name='block1_bn')(x)\n",
    "    # Block 2\n",
    "    x = Conv1D(16, 3, activation='relu', padding='valid', name='block2_conv1')(x)\n",
    "    x = MaxPooling1D(2, name='block2_pool')(x)\n",
    "    # Block 3\n",
    "    x = Conv1D(32, 3, activation='relu', padding='valid', name='block3_conv1')(x)\n",
    "    x = MaxPooling1D(2, name='block3_pool')(x)\n",
    "    x = BatchNormalization(name='block3_bn')(x)\n",
    "    # Block 4\n",
    "    x = Conv1D(64, 3, activation='relu', padding='valid', name='block4_conv1')(x)\n",
    "    x = MaxPooling1D(2, name='block4_pool')(x)\n",
    "    # Block 5\n",
    "    x = Conv1D(64, 3, activation='relu', padding='valid', name='block5_conv1')(x)\n",
    "    x = MaxPooling1D(2, name='block5_pool')(x)\n",
    "    x = BatchNormalization(name='block5_bn')(x)\n",
    "    # Block 6\n",
    "    x = Conv1D(256, 3, activation='relu', padding='valid', name='block6_conv1')(x)\n",
    "    x = MaxPooling1D(2, name='block6_pool')(x)\n",
    "    x = Dropout(drop_out)(x)\n",
    "    # Classification block\n",
    "    x = Flatten(name='flatten')(x)\n",
    "    x = Dense(50, activation='relu', name='fc1')(x)\n",
    "    x = Dense(50, activation='relu', name='fc2')(x)\n",
    "    x = Dropout(drop_out)(x)\n",
    "    x = Dense(classes, activation='softmax', name='predictions')(x)\n",
    "\n",
    "    inputs = img_input\n",
    "    # Create model.\n",
    "    model = Model(inputs, x, name='cnn')\n",
    "    optimizer = Adam(lr=LEARNING_RATE) # this one had 0.001\n",
    "    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])\n",
    "    return model\n",
    "\n",
    "\n",
    "### CNN model 4 - VGG16 based from: https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d\n",
    "def cnn4(classes=3):\n",
    "    # From VGG16 design\n",
    "    input_shape = (SAMPLE_HIGH,1)\n",
    "    img_input = Input(shape=input_shape)\n",
    "    # Block 0\n",
    "    x = BatchNormalization()(img_input)\n",
    "    \n",
    "    # Block 1\n",
    "    x = Conv1D(32, 3, activation='relu', padding='valid', name='block1_conv1')(x)\n",
    "    x = MaxPooling1D(2, name='block1_pool')(x)\n",
    "    x = Dropout(0.25)(x)\n",
    "    \n",
    "    # Block 2\n",
    "    x = Conv1D(64, 3, activation='relu', padding='valid', name='block2_conv1')(x)\n",
    "    x = MaxPooling1D(2, name='block2_pool')(x)\n",
    "    x = Dropout(0.25)(x)\n",
    "\n",
    "    # Block 3\n",
    "    x = Conv1D(128, 3, activation='relu', padding='valid', name='block3_conv1')(x)\n",
    "    x = Dropout(0.4)(x)\n",
    "    \n",
    "    # Classification block\n",
    "    x = Flatten(name='flatten')(x)\n",
    "    x = Dense(128, activation='relu', name='fc1')(x)\n",
    "\n",
    "    x = Dropout(0.3)(x)\n",
    "    x = Dense(classes, activation='softmax', name='predictions')(x)\n",
    "\n",
    "    inputs = img_input\n",
    "    # Create model.\n",
    "    model = Model(inputs, x, name='cnn')\n",
    "    optimizer = Adam(lr=LEARNING_RATE) # this one had 0.001\n",
    "    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])\n",
    "    return model\n",
    "\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
