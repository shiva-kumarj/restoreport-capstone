"""
train_model.py

This script trains a Balanced Random Forest Classifier model using pre-trained GloVe word embeddings
and a dataset of annotated reviews. The trained model is then saved to disk for future use.

The script performs the following steps:

1. Loads the annotated reviews dataset from a CSV file.
2. Tokenizes the text data into individual words.
3. Maps each word to its corresponding pre-trained GloVe word embedding vector.
4. Aggregates the word embeddings for each review by averaging them.
5. Splits the data into training and testing sets.
6. Initializes a Balanced Random Forest Classifier with custom class weights to handle class imbalance.
7. Trains the classifier on the training data.
8. Evaluates the model's performance on the test data and prints the accuracy and classification report.
9. Saves the trained model to disk as a pickle file.

The script assumes the following directory structure:

root_dir/
    data/
        processed/
            reviews/
                annotated_sample/
                    combined_annotated_reviews/
                        combined_annotated_reviews.csv
    ml-model-building/
        model/
            brf_text_classifier.pkl

The required files are:
- annotated_reviews.csv: The dataset containing the annotated reviews.
- glove.6B.300d.txt: The pre-trained GloVe word embeddings file.

The trained model is saved as 'brf_text_classifier.pkl' in the 'model' directory.

Usage:
1. Ensure that the required files and directory structure are in place.
2. Run the script: `python train_model.py`

The script will print the model's accuracy and classification report on the test data.
The trained model will be saved to the specified path ('brf_text_classifier.pkl').
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle


# map words to embeddings
def map_word_to_embeddings(word):
    try:
        return word_vectors[word]
    except KeyError:
        return np.zeros(300)


# aggregate word embeddings for each review
# Using averaging
# use aggregated word embeddings as features
def save_model(model, checkpoint_path):
    with open(checkpoint_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    root_dir = r"D:\My-Projects\stonecap"
    data_dir = os.path.join(root_dir, "data")
    processed_data_dir = os.path.join(data_dir, "processed")
    processed_reviews_dir = os.path.join(processed_data_dir, "reviews")
    annotated_reviews_dir = os.path.join(
        processed_reviews_dir,
        "annotated_sample",
        "combined_reviews",
        "combined_reviews.csv",
    )

    # Set model save path
    model_path = os.path.join(root_dir, "ml-model", "model", "brf_text_classifier.pkl")

    # Load pretrained work embeddings
    glove_file = r"D:\My-Projects\stonecap\glove.6B\glove.6B.300d.txt"
    word_vectors = KeyedVectors.load_word2vec_format(glove_file, binary=False)

    # load your dataset into pandas
    data = pd.read_csv(annotated_reviews_dir)

    # tokenize text data into individual words
    tokenized_texts = data["text"].apply(lambda x: x.split())

    mapped_texts = [
        [map_word_to_embeddings(word) for word in review] for review in tokenized_texts
    ]

    averaged_embeddings = [np.mean(review, axis=0) for review in mapped_texts]

    # convert the list of arrays into a numpy array
    X = np.array(averaged_embeddings)
    print(f"X shape: {X.shape}")
    # prepare target variable
    y = data["db_label"]

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=43
    )

    # Define custom class weights for each class to address class imbalance
    class_weights = {
        "ambience": 1,
        "food": 0.05,
        "order": 2,
        "service": 0.5,
        "staff": 2,
        "time": 2,
    }
    # Initialize Balanced Random Forest classifier
    brf_classifier = BalancedRandomForestClassifier(class_weight=class_weights)

    # Train the classifier
    brf_classifier.fit(X_train, y_train)

    # Predict labels for test data
    brf_y_pred = brf_classifier.predict(X_test)
    brf_train_pred = brf_classifier.predict(X_train)

    # Calculate accuracy
    brf_accuracy = accuracy_score(y_test, brf_y_pred)
    training_accuracy = accuracy_score(y_train, brf_train_pred)
    print("Balanced Random Forest Accuracy:", brf_accuracy)
    print("Balanced Random Forest Training Accuracy:", training_accuracy)

    print(classification_report(y_test, brf_y_pred))

    # Saving trained model to dick
    save_model(brf_classifier, model_path)
