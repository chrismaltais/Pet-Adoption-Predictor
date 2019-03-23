import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
if __name__ == "__main__":
    # read data from csv
    x_train = pd.read_csv('x_train.csv', index_col=0)
    y_train = pd.read_csv('y_train.csv', index_col=0)
    x_test = pd.read_csv('x_test.csv', index_col=0)
    y_test = pd.read_csv('y_test.csv', index_col=0)
    # instantiate model
    model = Sequential()
    # add hidden layer with 32 nodes and soft-sign activation
    model.add(Dense(100, input_dim=x_train.iloc[0].size))
    model.add(Activation('softsign'))
    # add output layer with 5 nodes and soft-max activation
    model.add(Dense(5))
    model.add(Activation('softmax'))
    # use Adaptive Subgradient learning method with Categorical Crossentropy loss function
    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    # convert y values to one-hot vectors
    y_train_oh = to_categorical(y_train.values, num_classes=5)
    y_test_oh = to_categorical(y_test.values, num_classes=5)
    # train the model
    model.fit(x_train.values, y_train_oh, epochs=10, batch_size=1)
    print(model.evaluate(x_test.values, y_test_oh))
    print(model.predict(x_test.values))