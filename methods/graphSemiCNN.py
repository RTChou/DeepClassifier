import keras.layers.convolutional as conv
import keras.layers.pooling as pool
import keras.regularizers as reg
from keras.layers import Input, Activation, Flatten, Dense, Concatenate, Reshape
from keras.layers.merge import Dot
from keras.models import Model
from keras.optimizers import SGD # stochastic gradient descent

"""
Semi-supervised learning with convolution neural network
"""
class GraphSemiCNN:
    def __init__(self, trainX, trainY, testX, testY, nb_classes, nb_epochs=75, batch_size=64):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        
        self.input_samples = trainX.shape[0]
        self.input_genes = trainX.shape[1]
        
        self.nb_filters = 1000
        self.kernel_size = 10
        self.L1CNN = 0
        self.actfun = 'relu'
        self.pool_size = 11 # window size for features
        
        self.units1 = 11000 # number of nodes in hidden layer
        self.units2 = 5500
        self.units3 = 5500
        self.units4 = 5500
       
        # self.dropout = 0.75 # param for preventing overfitting

        self.nb_classes = nb_classes
        self.INIT_LR = 0.01 # initial learning rate
        self.lambd = 1.0
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
    
    def build(self):
        input1 = Input(shape=(self.input_genes, 1))
        feature1 = conv.Conv1D(self.nb_filters, self.kernel_size, padding='same', kernel_initializer='he_normal', 
                kernel_regularizer=reg.l1(self.L1CNN))(input1)
        feature1 = Activation(self.actfun)(feature1)
        feature1 = pool.MaxPooling1D(self.pool_size)(feature1)
        feature1 = Flatten()(feature1)
    
        input2 = Input(shape=(self.input_genes, 1))
        feature2 = conv.Conv1D(self.nb_filters, self.kernel_size, padding='same', kernel_initializer='he_normal', 
                kernel_regularizer=reg.l1(self.L1CNN))(input2)
        feature2 = Activation(self.actfun)(feature2)
        feature2 = pool.MaxPooling1D(self.pool_size)(feature2)
        feature2 = Flatten()(feature2)
        
        hidden1_1 = Dense(self.units1, activation='relu')(feature1)
        target = Dense(self.units3, activation='relu')(hidden1_1) # z1 -> z3
        hidden1_2 = Dense(self.units1, activation='relu')(feature2)
        context = Dense(self.units3, activation='relu')(hidden1_2)
        hidden2 = Dense(self.units2, activation='relu')(hidden1_1) # z1 -> z2
        hidden4 = Dense(self.units4, activation='relu')(target) # z3 -> z4
        concatenated = Concatenate(axis=0)([hidden2, hidden4]) # concatenate z2, z4
    
        dot_product = Dot(axes=1)([target, context])
        dot_product = Reshape((1,))(dot_product)
        
        output1 = Dense(self.nb_classes, activation='softmax', name='output1')(concatenated)
        output2 = Dense(1, activation='sigmoid', name='output2')(dot_product) # softmax?
    
        self.model = Model(inputs=[input1, input2], outputs=[output1, output2], name='graphSemiCNN')
    
        losses = {
            'output1': 'categorical_crossentropy',
            'output2': 'binary_crossentropy',
        }
        lossWeights = {'output1': 1.0, 'output2': self.lambd}  
        opt = SGD(lr=self.INIT_LR)
        
        cnn = self.model.compile(loss=losses, loss_weights=lossWeights, optimizer=opt, metrics=['accuracy']) # loss function: cross entropy
        return cnn

    def fit(self):
        fitHistory = self.model.fit(self.trainX, self.trainY, validation_data=(self.testX, self.testY), 
                epochs=self.nb_epochs, batch_size=self.batch_size)
        return fitHistory

    def predict(self):
        predictions = self.model.predict(self.testX, self.batch_size)
        return predictions

