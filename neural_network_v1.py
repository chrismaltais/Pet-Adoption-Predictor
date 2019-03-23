import os
import json
import pandas as pd
import numpy as np
# Import the model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
from matplotlib import pyplot as plt

# Suppress Warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Returns Dataframe of Pet_ID, magnitude, score
def get_sentiment_values(folderpath):
    data = []
    
    # df = pd.DataFrame(index=columns[0], columns=columns)
    for filename in os.listdir(folderpath):
        # Get magnitude and score
        with open(folderpath + '/' + filename) as json_file:
            contents = json.load(json_file)
            magnitude = contents["documentSentiment"]["magnitude"]
            score = contents["documentSentiment"]["score"]
        row = []
        # Get PetID
        filename_split = filename.split(".")
        petID = filename_split[0]
        row.append(petID)
        row.append(magnitude)
        row.append(score)
        data.append(row)
    # Convert to DataFrame
    columns = ['PetID', 'Magnitude', 'Score']
    results = pd.DataFrame(data=data, columns=columns)
    results.set_index('PetID', inplace=True)
    return results
    
if __name__ == "__main__":
    # Get Training Data
    x_train = pd.read_csv('x_train.csv', index_col=0)
    y_train = pd.read_csv('y_train.csv', index_col=0)
    
    # Get Testing Data
    x_test = pd.read_csv('x_test.csv', index_col=0)
    y_test = pd.read_csv('y_test.csv', index_col=0)
    
    # Find number of each type of adoption
    # print(len(dataset_train[dataset_train.AdoptionSpeed == 0]))

    # Show correlation table
    # plt.matshow(x_train.corr())
    # plt.show()

    # Extract sentiment from JSON files
    train_sentiment = get_sentiment_values('data/train_sentiment')
    
    # Add sentiment to train and test data
    x_train = x_train.merge(train_sentiment, on='PetID', how='left')
    x_test = x_test.merge(train_sentiment, on='PetID', how='left')

    x_train.replace(to_replace=float('NaN'), value=0, inplace=True)
    x_test.replace(to_replace=float('NaN'), value=0, inplace=True)

    # Initializing the multilayer perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(10), solver='sgd', learning_rate_init= 0.01, max_iter=500)

    # Fit the model
    mlp.fit(x_train, y_train)

    predicted_results = mlp.predict(x_test)

    print("Actual Values: ", y_test['AdoptionSpeed'].values)
    
    print("Predicted Values: ", predicted_results)

    print("Score: ", mlp.score(x_test, y_test))

    # Confusion Matrix
    confusion_matrix_results = confusion_matrix(y_test.values, predicted_results)
    print(confusion_matrix_results)

    # Classification Report
    class_report_results = classification_report(y_test.values, predicted_results)
    print(class_report_results)

    # Kappa Score
    print('Quadratic Weighted Kappa Score: %0.3f' % cohen_kappa_score(y_test.values, predicted_results, weights = 'quadratic'))

    # Number 4 = 4197
    # Number 3 = 3259
    # Number 2 = 4037
    # Number 1 = 3090
    # Number 0 = 410

    # PCA
    # Correlation Matrix

    # Not normalize because data is categorical

    # Handling JSON Data:
    # Create dataframe w magnitude and score and pet_id
    # Every row might not have 











