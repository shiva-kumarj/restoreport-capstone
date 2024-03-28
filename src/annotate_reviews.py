"""
annotate_reviews.py

This script annotates a set of validated Yelp reviews with predicted labels using a pre-trained text classification model.
The annotated reviews are saved as CSV files in the 'output_dir' directory.

The script performs the following steps:

1. Loads the pre-trained text classification model from a specified path.
2. Loads the pre-trained GloVe word embeddings.
3. Iterates through a directory containing validated review CSV files.
4. For each file:
   a. Tokenizes the review text into individual words.
   b. Maps each word to its corresponding GloVe word embedding vector.
   c. Aggregates the word embeddings for each review by averaging them.
   d. Applies the pre-trained text classification model to predict labels for the reviews.
   e. Adds the predicted labels to the review data as a new column.
   f. Saves the annotated reviews as a new CSV file in the 'output_dir' directory.

The required inputs are:
- Path to the pre-trained text classification model.
- Path to the pre-trained GloVe word embeddings file.
- Path to the directory containing validated review CSV files.

The output directory is specified as 'output_dir'.

Usage:
1. Ensure that the required inputs (model path, word embeddings file, and validated review files) are available.
2. Run the script: `python annotate_reviews.py`

The script will annotate the validated reviews with predicted labels and save the annotated data as CSV files in the 'output_dir' directory.

Note: The script uses the tqdm library to display a progress bar during the annotation process.
"""

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import os
import pickle
from tqdm import tqdm

root_dir = r"D:\My-Projects\stonecap"
log_file_path = os.path.join(root_dir, "logs", "review_labelling.log")


# map words to embeddings
def map_word_to_embeddings(word):
    try:
        return word_vectors[word]
    except KeyError:
        return np.zeros(300)


def get_csv_filenames(directory):
    filenames = []
    for filename in os.listdir(directory):
        if "csv" in filename:
            filenames.append(filename)
    return filenames


def load_model(checkpoint_path):
    try:
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return "Model Not Found at specified path."


if __name__ == "__main__":
    root_dir = r"D:\My-Projects\stonecap"
    output_dir = os.path.join(
        root_dir, "data", "processed", "reviews", "labelled_dataset"
    )
    # glove_file = r"D:\My-Projects\glove.6B\glove.6B.300d.txt"
    glove_file = os.path.join(root_dir, "glove.6B", "glove.6B.300d.txt")

    # path to the saved model
    model_path = os.path.join(root_dir, "ml-model", "model", "brf_text_classifier.pkl")

    # Load model
    model = load_model(model_path)

    # Load pretrained work embeddings
    word_vectors = KeyedVectors.load_word2vec_format(glove_file, binary=False)

    # path to validated data
    validated_file_path = os.path.join(
        root_dir, "data", "processed", "reviews", "validated_data"
    )

    # list of files in the directory
    filenames = get_csv_filenames(validated_file_path)

    for file in tqdm(filenames, total=len(filenames), desc=f"Inferencing: "):
        data = pd.read_csv(os.path.join(validated_file_path, file))

        # tokenize text data into individual words
        tokenized_texts = data["text"].apply(lambda x: x.split())

        mapped_texts = [
            [map_word_to_embeddings(word) for word in review]
            for review in tokenized_texts
        ]

        # aggregate word embeddings for each review
        # Using averaging
        # use aggregated word embeddings as features

        averaged_embeddings = [np.mean(review, axis=0) for review in mapped_texts]

        # convert the list of arrays into a numpy array
        X = np.array(averaged_embeddings)
        y_pred = model.predict(X)
        data["predicted_label"] = y_pred
        data.to_csv(os.path.join(output_dir, file), index=False)
