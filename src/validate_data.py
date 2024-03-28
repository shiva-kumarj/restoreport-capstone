"""
validate_data.py

This script validates processed Yelp review data by checking for missing values, data type mismatches,
and other inconsistencies. It reads CSV files from a specified input directory, performs various validation
checks, removes invalid or inconsistent data, and saves the validated data as new CSV files in an output
directory.

The script performs the following validation checks:

1. Removes reviews with zero-length text.
2. Checks if the 'date' column has no null values and is of the correct data type (datetime64[ns]).
3. Ensures that the month and day values in the 'date' column are within the valid range.
4. Checks if the 'stars' and 'useful' columns have no null values and are of the correct data type (int64).
5. Ensures that the 'stars' values are within the valid range of 1 to 5.
6. Checks if the 'business_id' column has no null values.
7. Ensures that there are no null values in the entire data chunk.

The script reads the processed review CSV files from the 'processed_reviews_dir' directory and writes
the validated data to the 'output_dir' directory.

Usage:
1. Ensure that the required input directory ('processed_reviews_dir') exists and contains the processed
   review CSV files.
2. Run the script: `python validate_data.py`

The script will validate the data, remove invalid or inconsistent entries, and save the validated data as
CSV files in the 'output_dir' directory.

Note: The script uses the Pandas library for data manipulation and validation.
"""

import pandas as pd
import os


def validate_text(chunk):
    """Check length of tokenized review. some reviews are zero length."""
    assert chunk["text"].apply(lambda x: len(x.strip())).min() > 0


def validate_datetime(chunk):
    assert (
        not chunk["date"].isnull().any()
    ), 'AssertionError: Null values found in the "date" column'
    assert (
        chunk["date"].dtype == "datetime64[ns]"
    ), "AssertionError: dtype mismatch of date column"
    assert not (
        (chunk["date"].dt.month > 12) | (chunk["date"].dt.month < 1)
    ).any(), "AssertionError: Month should be between 1 and 12 (inclusive)"
    assert not (
        (chunk["date"].dt.day > 31) | (chunk["date"].dt.day < 1)
    ).any(), "AssertionError: Date should be between 1 and 31 (inclusive)"


def validate_numerical_col(chunk):
    numerical_col = ["stars", "useful"]
    for col in numerical_col:
        assert (
            chunk[col].dtype == "int64"
        ), f"AssertionError: {col} should have data type 'int64'"
        assert (
            not chunk[col].isnull().any()
        ), f"AssertionError: {col} should not have missing values"

    assert (
        1.0 <= chunk["stars"].min() <= 5.0
    ), "AssertionError: 'stars' should be in the range of 1 to 5"
    assert (
        1.0 <= chunk["stars"].max() <= 5.0
    ), "AssertionError: 'stars' should be in the range of 1 to 5"

    return True


def validate_data(chunk):
    assert (
        not chunk["business_id"].isnull().any()
    ), "AssertionError: 'Business id' should not have missing values"
    validate_numerical_col(chunk)
    validate_datetime(chunk)
    assert (
        not chunk.isnull().any().any()
    ), "AssertionError: Chunk must not contain any missing values"
    return True


def get_csv_filenames(directory):
    filenames = []
    for filename in os.listdir(directory):
        if "csv" in filename:
            filenames.append(filename)
    return filenames


if __name__ == "__main__":
    root_dir = r"D:\My-Projects\stonecap"
    data_dir = os.path.join(root_dir, "data")
    processed_data_dir = os.path.join(data_dir, "processed")
    processed_reviews_dir = os.path.join(
        processed_data_dir, "reviews", "sentiment_analysis"
    )
    output_dir = os.path.join(processed_data_dir, "reviews", "validated_data")

    filenames = get_csv_filenames(processed_reviews_dir)
    input_file_paths = [os.path.join(processed_reviews_dir, file) for file in filenames]

    for file in input_file_paths:
        # Load processed review csv datasets
        data = pd.read_csv(file, parse_dates=["date"])
        before = len(data)

        # Filter based on which text reviews are not of string type.
        not_string_indices = data[
            data["text"].apply(lambda x: isinstance(x, str)) == False
        ].index

        # drop entries with non string reviews
        data = data.drop(not_string_indices)

        # Filter to check if review text is Zero-length
        zero_length_sentence_indices = data[
            data["text"].apply(lambda x: len(x.split())) == 0
        ].index

        # drop entries with Zero length review text
        data = data.drop(zero_length_sentence_indices)
        after = len(data)

        # extract file x
        file_name = file.split("\\")[-1]
        chunk_number = file_name[file_name.find("_") + 1 : file_name.find(".")]

        print(f"Dropped {before-after} rows from Chunk_{chunk_number}")

        # Data validation Suite.
        validate_data(data)

        data.to_csv(os.path.join(output_dir, f"chunk_{chunk_number}.csv"))
