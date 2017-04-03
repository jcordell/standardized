import KerasDeepFeedFoward as nn
import configuration_parser
import numpy as np
import os

class model():
    def  __init__(self, epochs, learning_rate, hidden1, hidden2, hidden3, savepath, batch_size, num_ensemble):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.savepath = savepath
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble;


    def predict(self,x_test):
        # using ensemble prediction
        for i in range(0, self.num_ensemble):
            # unnormalize prediction
            ypredict = nn.predict(x_test, savepath = self.savepath, num_ensemble=i) * 758.92 #todo way to not hardcode this?
            if (i == 0):
                predictions = np.array(ypredict)
            else:
                predictions = np.column_stack((predictions, ypredict))
        return np.mean(np.array(predictions), axis=1)

    def fit(self, x_train, y_train):
        # normalize y data
        #y_train_norm = (y_train - min(y_train))/(max(y_train) - min(y_train))
        y_train_norm = y_train/758.92 #todo better way to normalize?
        # check if model folder already has trained models, skip training if true
        if os.listdir(self.savepath[:-3]) == ["todo"]: #todo
            print("model already fitted, returning")
            return
        for i in range(0, self.num_ensemble):
            nn.fit(x_train, y_train_norm, learning_rate=self.learning_rate, batch_size=self.batch_size,
                   hidden1=self.hidden1, hidden2 = self.hidden2, hidden3=self.hidden3,
                   epochs=self.epochs, savepath=self.savepath, num_ensemble = i)



def get():
    config = configuration_parser.parse()

    epochs = config.getint(__name__, 'epochs')
    savepath = config.get(__name__, 'savepath')
    hidden1 = config.get(__name__, 'hidden1')
    hidden2 = config.getint(__name__, 'hidden2')
    hidden3 = config.getint(__name__, 'hidden3')
    learning_rate = config.getfloat(__name__, 'learning_rate')
    batch_size = config.getint(__name__, 'batch_size')
    num_ensemble = config.getint(__name__, 'num_ensemble')

    return model(epochs, learning_rate, hidden1, hidden2, hidden3, savepath, batch_size, num_ensemble)
