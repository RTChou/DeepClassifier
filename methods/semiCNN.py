import keras.layers.convolutional as conv
import keras.layers.pooling as pool
import keras.regularizers as reg
from keras.layers import Input, Activation, Flatten, Dense, Concatenate
from keras.models import Model
from keras.optimizers import SGD # stochastic gradient descent

"""
Graph based semi-supervised learning with convolution neural network
"""
def GraphSemiCNN(trainX, trainY, testX, testY, nb_classes, predict=False):

    # initialization
    input_samples = trainX.shape[0]
    input_genes = trainX.shape[1]
    steps = 1
    filters = 1000 # N filters
    kernel_size = 10 # a window of size k
    L1CNN = 0
    # dropout = 0.75 # parm for preventing overfitting
    actfun = 'relu'
    pool_size = 11 # a window of p features
    units1 = 11000 # number of nodes in hidden layer
    units2 = 5500
    units3 = 5500
    units4 = 5500
    nb_classes = nb_classes # number of tissue types
    nb_nodes = input_samples # number of input samples
    INIT_LR = 0.01 # initial learning rate
    EPOCHS = 75 # number of epochs

    input = Input(shape=(input_samples, steps, input_genes))
    feature = conv.Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l1(L1CNN))(input)
    # initializer, regularizer, other params for conv
    # feature = Dropout(dropout)(feature)
    feature = Activation(actfun)(feature)
    feature = pool.MaxPooling1D(pool_size)(feature)
    feature = Flatten()(feature)
    
    hidden1 = Dense(units1, activation='relu')(feature)
    hidden2 = Dense(units2, activation='relu')(hidden1) # z1 -> z2
    hidden3 = Dense(units3, activation='relu')(hidden1) # z1 -> z3
    hidden4 = Dense(units4, activation='relu')(hidden3) # z3 -> z4
    concatenated = Concatenate(axis=0)([hidden2, hidden4]) # concatenate z2, z4
    
    output1 = Dense(nb_classes, activation='softmax')(concatenated)
    output2 = Dense(nb_nodes, activation='softmax')(hidden3)

    cnn = Model(input, [output1, output2])
    
    print('[INFO] training network...')
    opt = SGD(lr=INIT_LR)
    cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) # loss function: cross entropy
    
    # train the model
    if predict is False:
        H = cnn.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=64)

    return cnn

