import pandas as pd
import re
import numpy as np
import stanza
import os
import logging
from tqdm import tqdm

import checkpoint

tqdm.pandas()

log_file_path = os.path.abspath("D:\My-Projects\stonecap\logs\lemmatization.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


nlp = stanza.Pipeline(processors="tokenize,sentiment", lang="en", use_gpu=True)


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


def clean_text(inp_text):
    # lower case
    inp_text = inp_text.lower()
    # extract out all alphabet, numbers, and select special characters and join them back together with a 'space'.
    regex_pattern = r'[a-zA-Z0-9s!?." "]+'
    matched_substrings = re.findall(regex_pattern, inp_text)
    cleaned_text = "".join(matched_substrings)
    # replace all non alphabetic, space and period characters with a period
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s.]", ".", cleaned_text)

    return cleaned_text


def clean_stars(stars_column):
    # choice_of_ratings = [np.floor(stars_column.mean()), stars_column.mode(), stars_column.quantile(0.5)]
    choice_of_ratings = [3, 4, 5]
    stars_column = stars_column.fillna(np.random.choice(choice_of_ratings))
    stars_column = stars_column.astype("int")
    return stars_column


def clean_useful(useful_column):
    useful_column = useful_column.fillna(useful_column.mode())
    useful_column = useful_column.astype("int")
    return useful_column


def clean_date(date_column):
    date_column = pd.to_datetime(date_column)
    return date_column


def clean_data(chunk):
    chunk.loc[:, "business_id"] = chunk["business_id"].dropna()
    chunk.loc[:, "stars"] = clean_stars(chunk["stars"])
    chunk.loc[:, "useful"] = clean_useful(chunk["useful"])
    chunk.loc[:, "date"] = clean_date(chunk["date"])
    chunk.loc[:, "text"] = chunk["text"].apply(lambda x: clean_text(x))
    return chunk


# 1. ~~Spell correction (did not add any value, very time taking)~~
# 2. ~~(Sentence, Sentiment) map using Stanza.~~
# 3. ~~Flatten (sentence, sentiment) map into individual rows.~~
# 4. annotate sentence into one word "business area" using OpenAI or other libraries available on github.
#    1. Note: OpenAI is paid but gave the best results so far. better than manual labelling, semi-supervised labelling, and stanza.


# Retrieve the sentiment of each noun from a sentence
def get_line_sentiment(
    review,
):
    sentiment_map = {}
    for sentence in nlp(review).sentences:
        sentiment = sentence.sentiment
        sentiment_map[sentence.text] = sentiment
    # record_count[0] += 1
    # logging.info(f'get_line_sentiment: {record_count[0]}/{record_count[1]}.')
    return sentiment_map


def get_sentiment_noun(sentiment_dict):
    return sentiment_dict.keys()


def get_sentiment_value(sentiment_dict):
    return sentiment_dict.values()


def unpack_sentiment(df):
    df.loc[:, "statement"] = df["sentiment_dict"].apply(get_sentiment_noun)
    df.loc[:, "sentiment"] = df["sentiment_dict"].apply(get_sentiment_value)

    df = df.explode(["statement", "sentiment"])

    df.drop(["sentiment_dict", "text"], axis=1, inplace=True)

    return df


def process_chunk(chunk, i, business_df, checkpoint):
    # process the chunk
    restaurant_b_ids = list(business_df["business_id"].unique())
    chunk = chunk[["business_id", "stars", "useful", "text", "date"]]
    filtered_df = chunk[chunk["business_id"].isin(restaurant_b_ids)]

    cleaned_data = clean_data(filtered_df)

    batch_size = len(cleaned_data)
    record_count = [0, batch_size]
    logging.info(f"Chunk: {i} batch size: {batch_size}")

    # validate the data
    validate_data(cleaned_data)

    # cleaned_data = cleaned_data.assign(
    #     sentiment_dict=cleaned_data["text"].apply(
    #         lambda x: get_line_sentiment(
    #             x,
    #         )
    #     )
    # )

    cleaned_data = cleaned_data.assign(
        sentiment_dict=cleaned_data["text"].progress_apply(
            lambda x: get_line_sentiment(
                x,
            )
        )
    )

    cleaned_data = unpack_sentiment(cleaned_data)

    # validate the data once again
    validate_data(cleaned_data)

    # Write file to storage
    # may also write it into Yelp_db
    cleaned_data.to_csv(
        f"D:\My-Projects\stonecap\data\processed\\reviews\chunk_{i}.csv"
    )

    # save checkpoint
    # Saving how many chunks were processed
    checkpoint.save_checkpoint(i + 1)

    logging.info(f"Processed chunk {i}")


if __name__ == "__main__":
    chunksize = 10000
    # Process the chunks first
    business_df = pd.read_csv(r"D:\My-Projects\stonecap\data\processed\business.csv")
    df_iter = pd.read_json(
        r"D:\My-Projects\stonecap\data\raw\yelp_academic_dataset_review.json",
        lines=True,
        chunksize=chunksize,
        encoding="utf-8",
    )

    # Create checkpoint object
    checkpoint_obj = checkpoint.Checkpoint(
        host="localhost", db="yelp_db", user="root", password="root"
    )

    # create "checkpoint" table in "yelp_db"
    checkpoint_obj.create_checkpoint()

    # Load checkpoints from db
    last_checkpoint = checkpoint_obj.load_checkpoint()[-1][-1]

    if last_checkpoint == 0:
        for i, chunk in enumerate(df_iter):
            process_chunk(chunk, i, business_df, checkpoint_obj)

    else:
        df_iter = pd.read_json(
            r"D:\My-Projects\stonecap\data\raw\yelp_academic_dataset_review.json",
            lines=True,
            chunksize=chunksize,
            encoding="utf-8",
        )
        for i, chunk in enumerate(df_iter):
            if i < last_checkpoint:
                continue
            process_chunk(chunk, i, business_df, checkpoint_obj)
