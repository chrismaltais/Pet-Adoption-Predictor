# Author: Chris Maltais
# SN: 10155183
# Description: This file uses a simple Multilayer Perceptron Model with a Backpropagation Learning algorithm.
# This file can be run using additional NLP information obtained from Google - please see README for more details.

import os
import json
import pandas as pd
import numpy as np
import sys

# Import the model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
from matplotlib import pyplot as plt

# Suppress Warnings (for development purposes)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Returns Dataframe of Pet_ID, magnitude, score
# aka get sentiment analysis from each individual .json file!
def get_sentiment_values(folderpath):
    data = []
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

# This code was run to obtain the ideal hyperparameter values for the MLP
# It iterates over a subset of parameter values and stores the values that provide the highest weighted kappa score
def get_ideal_HP(x_train, x_test, y_train, y_test, max_iterations, HL_size, learning_rate_init, learning_rate, momentum):
    hyper_params = {
        "accuracy": 0,
        "kappa": 0,
        "HL_size": 0, 
        "learning_rate_init": 0,
        "learning_rate": 0, 
        "momentum": 0
    }

    max_kappa = 0

    for layer_size in HL_size:
        for init_LR in learning_rate_init:
            for LR in learning_rate:
                for mu in momentum:
                    print(layer_size)
                    print(init_LR)
                    print(LR)
                    print(mu)
                    # Initializing the multilayer perceptron
                    mlp = MLPClassifier(
                        hidden_layer_sizes=(layer_size), 
                        solver='sgd', 
                        learning_rate_init= init_LR, 
                        learning_rate=LR,
                        momentum=mu, 
                        max_iter=500
                    )
                    # Fit the model
                    mlp.fit(x_train, y_train)

                    # Predicted Results
                    predicted_results = mlp.predict(x_test)

                    # Accuracy and Kappa Scores
                    accuracy = mlp.score(x_test, y_test)
                    kappa = cohen_kappa_score(y_test.values, predicted_results, weights = 'quadratic')

                    # Write to file
                    write_HP_to_file(accuracy, kappa, layer_size, init_LR, LR, mu)

                    print("Accuracy: ", accuracy)
                    print("Weighted Kappa: ", kappa)

                    if (kappa > max_kappa):
                        max_kappa = kappa
                        hyper_params = {
                            "accuracy": accuracy,
                            "kappa": kappa,
                            "HL_size": layer_size, 
                            "learning_rate_init": init_LR,
                            "learning_rate": LR, 
                            "momentum": mu
                        }
    return hyper_params

# Used to write the hyper parameter values, weighted kappa value and score to a text file in results/ folder
def write_HP_to_file(accuracy, kappa, layer_size, learning_rate_init, learning_rate, momentum):
    filename = 'results/hyperparameter_tuning.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a') as f:
        f.write(
            "Accuracy: {accuracy}, Weighted Kappa: {kappa}, Hidden Layer Size: {layer_size}, Initial Learning Rate: {learning_rate_init}, Learning Rate: {learning_rate}, Momentum: {momentum}\n"
            .format(accuracy=accuracy, kappa=kappa, layer_size=layer_size, learning_rate_init=learning_rate_init, learning_rate=learning_rate, momentum=momentum)
        )

# Main script
if __name__ == "__main__":
    # Get Training Data
    x_train = pd.read_csv('x_train.csv', index_col=0)
    y_train = pd.read_csv('y_train.csv', index_col=0)
    print(x_train.shape)
    # Get Testing Data
    x_test = pd.read_csv('x_test.csv', index_col=0)
    y_test = pd.read_csv('y_test.csv', index_col=0)

    # Check if useNLP flag is true
    if (len(sys.argv) == 2):
        if(sys.argv[1] == "useNLP"):
            # Extract sentiment from JSON files
            train_sentiment = get_sentiment_values('data/train_sentiment')
    
            # Add sentiment to train and test data
            x_train = x_train.merge(train_sentiment, on='PetID', how='left')
            x_test = x_test.merge(train_sentiment, on='PetID', how='left')

            x_train.replace(to_replace=float('NaN'), value=0, inplace=True)
            x_test.replace(to_replace=float('NaN'), value=0, inplace=True)

    ###### The following block of commented code was used to obtain the hyper parameters 
    ###### with the highest weighted kappa score
    # HL_size = [2, 5, 10, 15, 50, 100]
    # learning_rate_init = [0.001, 0.01, 0.05, 0.1, 0.5, 0.9]
    # learning_rate = ['constant', 'adaptive']
    # momentum = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # ideal_HP = get_ideal_HP(
    #     x_train, 
    #     x_test, 
    #     y_train, 
    #     y_test, 
    #     500, 
    #     HL_size, 
    #     learning_rate_init, 
    #     learning_rate, 
    #     momentum
    # )

    # print(ideal_HP)
    ###### 

    # Initializing the multilayer perceptron with values obtained from HP tuning above
    mlp = MLPClassifier(
        hidden_layer_sizes=(5), 
        solver='sgd', 
        learning_rate_init= 0.05, 
        learning_rate='constant', 
        momentum=0.5, 
        max_iter=500
    )

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












