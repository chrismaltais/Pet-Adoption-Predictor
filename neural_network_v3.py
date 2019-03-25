import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score
    
def decode_output (y_onehot):
    ret = []
    for y in y_onehot:
        ret.append(np.argmax(y))
    return np.array(ret)

if __name__ == "__main__":
    # read data from csv
    x_train = pd.read_csv('x_train.csv', index_col=0)
    y_train = pd.read_csv('y_train.csv', index_col=0)
    x_test = pd.read_csv('x_test.csv', index_col=0)
    y_test = pd.read_csv('y_test.csv', index_col=0)
    # instantiate model
    model = Sequential()
    # add hidden layer with 32 nodes and soft-sign activation
    model.add(Dense(50, input_dim=x_train.iloc[0].size))
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
    model.fit(x_train.values, y_train_oh, epochs=50, batch_size=1)
    print(model.evaluate(x_test.values, y_test_oh))
    y_predict = model.predict(x_test)
    y_decode = decode_output(y_predict)
    print(cohen_kappa_score(y_decode, y_test))
