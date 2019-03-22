import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# convert categorical columns to one-hot binary arrays
def one_hot_encoder(df, cols_to_dum, nan_as_category = True):
    original_columns = list(df.columns)
    cat_columns = [col for col in cols_to_dum]
    df = pd.get_dummies(df, columns = cat_columns, dummy_na = False)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
    
if __name__ == "__main__":
    # Get Training Data
    dataset = pd.read_csv('../data/train.csv')
    columns_to_keep = ['Type', 'Age', 'Breed1','Breed2','Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee','State','PhotoAmt']
    cols_not_to_encode = ['Age','VideoAmt','PhotoAmt','Fee','Quantity']
    # get target values
    y_data = dataset['AdoptionSpeed']
    # drop unwanted columns
    x_data = dataset[columns_to_keep]
    cols_total = x_data.columns
    # encode categorical columns
    cols_to_encode = np.setdiff1d(cols_total,cols_not_to_encode)
    x_encoded, cat_cols = one_hot_encoder(x_data, cols_to_encode)
    # split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_data, test_size=0.20)

    # scale input data
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    model = Sequential()
    # add hidden layer with 32 nodes and soft-sign activation
    model.add(Dense(100, input_dim=len(x_train_std[0])))
    model.add(Activation('softsign'))
    # add output layer with 5 nodes and soft-max activation
    model.add(Dense(5))
    model.add(Activation('softmax'))
    # use Adaptive Subgradient learning method with Categorical Crossentropy loss function
    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    # convert y values to one-hot vectors
    y_train_oh = to_categorical(y_train, num_classes=5)
    y_test_oh = to_categorical(y_test, num_classes=5)
    # train the model
    model.fit(x_train_std, y_train_oh, epochs=10, batch_size=1)
    print(model.evaluate(x_test_std, y_test_oh))
    print(model.predict(x_test_std))