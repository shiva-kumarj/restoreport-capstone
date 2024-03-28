"""
review_labeller_huggingface.py

This Python script is designed to enrich a collection of reviews with labels using a pre-trained DeBERTa model for zero-shot text classification. The script reads in chunks of review data from CSV files, applies the DeBERTa model to classify each review into one of the predefined labels, and saves the enriched data back to CSV files.

The script utilizes the following components:

1. Concurrent Processing:
  - The `ThreadPoolExecutor` from the `concurrent.futures` module is used to process multiple chunks of data concurrently, taking advantage of multiple CPU cores for improved performance.

2. Progress Tracking:
  - The `tqdm` library is employed to display progress bars for the enrichment process, providing visual feedback on the progress of processing each chunk.

3. Logging:
  - The `logging` module is configured to log informational messages to a specified log file, facilitating debugging and monitoring the enrichment process.

4. Checkpointing:
  - The script implements a checkpointing mechanism to keep track of the last processed chunk. This allows the enrichment process to resume from the last checkpoint in case of interruptions or restarts.

5. Hugging Face Transformers:
  - The `transformers` library from Hugging Face is used to load and utilize the pre-trained DeBERTa model for zero-shot text classification.

The main functions in the script are:

- `deberta_classifier(row)`: Takes a row from the review data and applies the DeBERTa model to classify the review text into one of the predefined labels.
- `deberta_enricher(chunk_path, i)`: Reads a chunk of review data from a CSV file, applies the `deberta_classifier` function to each row, and saves the enriched data back to a new CSV file.
- `process_chunks(checkpoint_path)`: Orchestrates the enrichment process by iterating over the chunks of data, submitting tasks to the `ThreadPoolExecutor`, and updating the checkpoint file.
- `save_checkpoint(last_checkpoint, checkpoint_path)`: Saves the current checkpoint (index of the last processed chunk) to a file using pickle.
- `load_checkpoint(checkpoint_path)`: Loads the last checkpoint from the checkpoint file, or returns 0 if the file doesn't exist.

The script is designed to run on a Windows system and expects the data files to be located in specific directories. The paths to the data directories and the checkpoint file can be modified in the script as needed.

Usage:
1. Ensure that the required Python packages (e.g., pandas, transformers, tqdm) are installed.
2. Update the file paths in the script to match the locations of your data files and desired output directories.
3. Run the script: `python review_labeller_huggingface.py`

The script will start processing the chunks of review data, applying the DeBERTa model for label prediction, and saving the enriched data to the specified output directory. Progress updates and log messages will be displayed in the console and written to the specified log file.

Note: The script assumes that the review data is stored in CSV files, with each chunk named as "chunk_<index>.csv". The processed and enriched data will be saved in the "data/processed/reviews/annotated" directory.
"""

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
from transformers import pipeline
import os
import logging
import pickle

tqdm.pandas()
root_dir = r"D:\My-Projects\stonecap"

# log_file_path = os.path.abspath(r"D:\My-Projects\stonecap\logs\review_enrichment.log")
log_file_path = os.path.join(root_dir, "logs", "review_enrichment.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def deberta_classifier(row):
    candidate_labels = ["food", "time", "service", "ambience", "order", "staff"]

    classification_report = db_classifier(row["text"], candidate_labels)
    return classification_report["labels"][0]


def deberta_enricher(chunk_path, i):
    chunk = pd.read_csv(chunk_path, nrows=100)
    total_rows = len(chunk)
    # start_time = time.time()
    logging.info(f"Beginning Enrichment: Chunk {i} batch size: {len(chunk)}")

    for index, row in tqdm(
        chunk.iterrows(),
        desc=f"chunk_{i}",
        total=total_rows,
        unit="row",
        leave=False,
    ):
        result = deberta_classifier(row)
        chunk.at[index, "db_label"] = result

    file_path = os.path.join(
        root_dir,
        "processed",
        "reviews",
        "annotated_sample",
        os.path.basename(chunk_path),
    )
    chunk.to_csv(file_path)
    # end_time = time.time()
    logging.info(f"Enrichment Complete: {chunk_path}")


def process_chunks(checkpoint_path):
    last_checkpoint = load_checkpoint(checkpoint_path)

    chunk_paths = [
        os.path.join(
            root_dir,
            "data",
            "processed",
            "reviews",
            "validated_data",
            f"chunk_{i}.csv",
        )
        for i in range(last_checkpoint, 70)
    ]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(deberta_enricher, chunk_path, i)
            for i, chunk_path in enumerate(chunk_paths, start=last_checkpoint)
        ]
        for future in tqdm(futures, total=len(futures), desc="Processing Chunks"):
            future.result()

    save_checkpoint(last_checkpoint + len(chunk_paths), checkpoint_path)


def save_checkpoint(last_checkpoint, checkpoint_path):
    with open(checkpoint_path, "wb") as f:
        pickle.dump(last_checkpoint, f)


def load_checkpoint(checkpoint_path):
    try:
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return 0  # Start from the beginning if the checkpoint file doesn't exist


if __name__ == "__main__":
    checkpoint_path = os.path.join(
        root_dir,
        "checkpoints",
        "review_enrichment.pkl",
    )

    db_classifier = pipeline(
        "zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v1"
    )

    process_chunks(checkpoint_path)
