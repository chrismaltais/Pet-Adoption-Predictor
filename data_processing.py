import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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
    # set index of dataframe
    dataset_indexed = dataset.set_index('PetID')
    # get target values
    y_data = dataset_indexed['AdoptionSpeed']
    # drop unwanted columns
    columns_to_keep = ['Type', 'Age', 'Breed1','Breed2','Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee','State','PhotoAmt']
    cols_not_to_encode = ['Age', 'PhotoAmt', 'Fee', 'Quantity']
    x_data = dataset_indexed.loc[:, columns_to_keep]
    # normalize non-categorical columns
    cols = x_data.loc[:, cols_not_to_encode]
    x_data.loc[:, cols_not_to_encode] = cols.subtract(cols.mean()).divide(cols.std())
    # encode categorical columns
    cols_total = x_data.columns
    cols_to_encode = np.setdiff1d(cols_total,cols_not_to_encode)
    x_encoded, cat_cols = one_hot_encoder(x_data, cols_to_encode)
    # split x and y data on type to ensure equal proportion of cats and dogs in test/train sets
    x_t1 = x_encoded[x_encoded.Type_1 == 1]
    x_t2 = x_encoded[x_encoded.Type_2 == 1]
    y_t1 = y_data.loc[x_t1.index.values]
    y_t2 = y_data.loc[x_t2.index.values]
    # split data into train and test sets
    x_t1_train, x_t1_test, y_t1_train, y_t1_test = train_test_split(x_t1, y_t1, test_size=0.20, stratify=y_t1)
    x_t2_train, x_t2_test, y_t2_train, y_t2_test = train_test_split(x_t2, y_t2, test_size=0.20, stratify=y_t2)
    # join test and training sets
    x_train = x_t1_train.append(x_t2_train)
    x_test = x_t1_test.append(x_t2_test)
    y_train = y_t1_train.append(y_t2_train)
    y_test = y_t1_test.append(y_t2_test)
    # shuffle data sets so that data is not ordered by type
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    # write data sets to file
    x_train.to_csv('x_train.csv')
    x_test.to_csv('x_test.csv')
    y_train.to_csv(path_or_buf='y_train.csv', header=True)
    y_test.to_csv(path_or_buf='y_test.csv', header=True)