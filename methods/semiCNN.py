import keras.layers.convolutional as conv
import keras.layers.pooling as pool
import keras.regularizers as reg
from keras.layers import Input, Activation, Flatten, Dense, Concatenate, Reshape
from keras.models import Model
from keras.optimizers import SGD # stochastic gradient descent

"""
Semi-supervised learning with convolution neural network
"""
def build_model(trainX, trainY, testX, testY, nb_classes):

    # initialization
    input_samples = trainX.shape[0]
    input_genes = trainX.shape[1]
    filters = 1000
    kernel_size = 10
    L1CNN = 0
    # dropout = 0.75 # parm for preventing overfitting
    actfun = 'relu'
    pool_size = 11 # window of eatures
    units1 = 11000 # numberof nodes in hidden layer
    units2 = 5500
    units3 = 5500
    units4 = 5500
    INIT_LR = 0.01 # initial learning rate
    EPOCHS = 75

    input1 = Input(shape=(input_genes, 1))
    feature1 = conv.Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l1(L1CNN))(input1)
    feature1 = Activation(actfun)(feature1)
    feature1 = pool.MaxPooling1D(pool_size)(feature1)
    feature1 = Flatten()(feature1)

    input2 = Input(shape=(input_genes, 1))
    feature2 = conv.Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l1(L1CNN))(input2)
    feature2 = Activation(actfun)(feature2)
    feature2 = pool.MaxPooling1D(pool_size)(feature2)
    feature2 = Flatten()(feature2)
    
    hidden1_1 = Dense(units1, activation='relu')(feature1)
    target = Dense(units3, activation='relu')(hidden1_1) # z1 -> z3
    hidden1_2 = Dense(units1, activation='relu')(feature2)
    context = Dense(units3, activation='relu')(hidden1_2)
    hidden2 = Dense(units2, activation='relu')(hidden1_1) # z1 -> z2
    hidden4 = Dense(units4, activation='relu')(target) # z3 -> z4
    concatenated = Concatenate(axis=0)([hidden2, hidden4]) # concatenate z2, z4
    
    similarity = merge([target, context], mode='cos', dot_axes=0)
    dot_product = merge([target, context], mode='dot', dot_axes=1)
    dot_product = Reshape((1,))(dot_product)

    output1 = Dense(nb_classes, activation='softmax')(concatenated)
    output2 = Dense(1, activation='softmax')(dot_product) # sigmoid

    cnn = Model([input1, input2], [output1, output2])

    opt = SGD(lr=INIT_LR)
    cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) # loss function: cross entropy
    
    return cnn

