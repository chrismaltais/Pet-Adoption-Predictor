import pandas as pd
import numpy as np
# Import the model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
    
if __name__ == "__main__":
    # Get Training Data
    dataset_train = pd.read_csv('data/train.csv')
    y_value_train = dataset_train['AdoptionSpeed']
    # Train based on: Type, Age, Breed1, Breed2, Gender, Color1, Color2, Color3
    columns_to_drop = ['Name', 'State', 'RescuerID', 'VideoAmt', 'Description', 'PetID', 'PhotoAmt', 'AdoptionSpeed']
    x_value_train = dataset_train.drop(columns_to_drop, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_value_train, y_value_train, test_size=0.20)

    sc = StandardScaler()
    sc.fit(x_train)

    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    # To Do:
    # 1) Normalize lol
    # 2) 

    # Initializing the multilayer perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(2),solver='sgd',learning_rate_init= 0.01, max_iter=500)

    # Fit the model
    mlp.fit(x_train_std, y_train)

    print(mlp.predict(x_test_std))

    # Get Score
    print(mlp.score(x_test_std, y_test))










