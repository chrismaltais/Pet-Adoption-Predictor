import pandas as pd
import numpy as np
# Import the model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib import pyplot as plt
    
if __name__ == "__main__":
    # Get Training Data
    dataset_train = pd.read_csv('data/train.csv')
    y_value_train = dataset_train['AdoptionSpeed']
    # Train based on: Type, Age, Breed1, Breed2, Gender, Color1, Color2, Color3
    columns_to_drop = ['Name', 'State', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'] #PhotoAmt, VideoAmt
    x_value_train = dataset_train.drop(columns_to_drop, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_value_train, y_value_train, test_size=0.20, stratify=y_value_train)
    
    # Find number of each type of adoption
    # print(len(dataset_train[dataset_train.AdoptionSpeed == 0]))

    plt.matshow(x_value_train.corr())
    plt.show()


    # sc = StandardScaler()
    # sc.fit(x_train)

    # x_train_std = sc.transform(x_train)
    # x_test_std = sc.transform(x_test)

    # Initializing the multilayer perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(2), solver='sgd', learning_rate_init= 0.01, max_iter=500)

    # Fit the model
    # mlp.fit(x_train_std, y_train)
    mlp.fit(x_train, y_train)

    predicted_results = mlp.predict(x_test)

    print("Actual Values: ", y_test.values)
    
    print("Predicted Values: ", predicted_results)

    print(mlp.score(x_test, y_test))

    # Confusion Matrix
    confusion_matrix_results = confusion_matrix(y_test.values, predicted_results)
    print(confusion_matrix_results)

    # Classification Report
    class_report_results = classification_report(y_test.values, predicted_results)
    print(class_report_results)

    # Get Score
    #print(mlp.score(x_test_std, y_test))

    # Number 4 = 4197
    # Number 3 = 3259
    # Number 2 = 4037
    # Number 1 = 3090
    # Number 0 = 410

    # Add stratification
    # Duplicate 0 rows 
    # Truncate others 

    # PCA
    # Correlation Matrix

    # Not normalize because data is categorical

    # Handling JSON Data:
    # Create dataframe w magnitude and score and pet_id
    # Every row might not have 

    # Questions to Prof:
    # Average
    # One final big model
    # Take each NN as an input and get result











