import pandas as pd
import re
import numpy as np
import os
import logging
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor
from textblob import TextBlob

root_dir = r"D:\My-Projects\stonecap"
data_dir = os.path.join(root_dir, "data")
processed_data_dir = os.path.join(data_dir, "processed")
processed_reviews_dir = os.path.join(processed_data_dir, "reviews")
checkpoints_dir = os.path.join(root_dir, "checkpoints")

tqdm.pandas()
log_file_path = os.path.join(root_dir, "logs", "full_review_processing.log")

# log_file_path = os.path.abspath(
#     r"D:\My-Projects\stonecap\logs\full_review_processing.log"
# )

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def save_checkpoint(last_checkpoint, checkpoint_path):
    with open(checkpoint_path, "wb") as f:
        pickle.dump(last_checkpoint, f)


def load_checkpoint(checkpoint_path):
    try:
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return 0  # Start from the beginning if the checkpoint file doesn't exist


def clean_text(inp_text):
    # lower case
    inp_text = inp_text.lower()
    # extract out all alphabet, numbers, and select special characters and join them back together with a 'space'.
    regex_pattern = r'[a-zA-Z0-9" "]+'
    matched_substrings = re.findall(regex_pattern, inp_text)
    cleaned_text = "".join(matched_substrings)
    # replace all non alphabetic, space and period characters with a period
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s.]", "", cleaned_text)

    return cleaned_text


def clean_stars(stars_column):
    # choice_of_ratings = [np.floor(stars_column.mean()), stars_column.mode(), stars_column.quantile(0.5)]
    choice_of_ratings = [3, 4]
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


def get_sentiment_textblob(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    # Classify sentiment based on polarity score
    if sentiment_score > 0.5:
        sentiment = "positive"
    elif sentiment_score < 0.25:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return sentiment


def process_chunk(chunk, i, business_df):
    # process the chunk
    restaurant_b_ids = list(business_df["business_id"].unique())
    chunk = chunk[["business_id", "stars", "useful", "text", "date"]]
    filtered_df = chunk[chunk["business_id"].isin(restaurant_b_ids)]

    cleaned_data = clean_data(filtered_df)

    batch_size = len(cleaned_data)
    logging.info(f"Chunk: {i} batch size: {batch_size}")

    # validate the data
    # validate_data(cleaned_data)
    cleaned_data = cleaned_data.assign(
        sentiment=cleaned_data["text"].progress_apply(
            lambda x: get_sentiment_textblob(
                x,
            )
        )
    )

    # Write file to storage
    # may also write it into Yelp_db
    output_file_path = os.path.join(
        processed_reviews_dir, "sentiment_analysis", f"chunk_{i}.csv"
    )

    cleaned_data.to_csv(output_file_path)

    logging.info(f"Processed chunk {i}")


def process_chunk_parallel(args):
    chunk, i, business_df, checkpoint_path = args
    process_chunk(chunk, i, business_df)
    # Save checkpoint after processing each chunk
    save_checkpoint(i + 1, checkpoint_path)


if __name__ == "__main__":
    chunksize = 100000
    # path to the checkpoints file
    checkpoint_path = os.path.join(checkpoints_dir, "review_processing_checkpoint.pkl")

    # Process the chunks first
    last_checkpoint = load_checkpoint(checkpoint_path)
    business_df_path = os.path.join(processed_data_dir, "business", "business.csv")
    business_df = pd.read_csv(business_df_path)

    raw_reviews_path = os.path.join(
        root_dir, "data", "raw", "yelp_academic_dataset_review.json"
    )
    df_iter = pd.read_json(
        raw_reviews_path,
        lines=True,
        chunksize=chunksize,
        encoding="utf-8",
    )

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Prepare arguments for parallel processing
        args_list = [
            (chunk, i, business_df, checkpoint_path)
            for i, chunk in enumerate(df_iter)
            if i >= last_checkpoint
        ]

        # Use ThreadPoolExecutor to process chunks in parallel
        for _ in tqdm(
            executor.map(process_chunk_parallel, args_list),
            total=len(args_list),
            desc="Processing Chunks",
        ):
            pass

        # save_checkpoint(len(df_iter))
