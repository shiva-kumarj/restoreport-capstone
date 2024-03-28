"""
combine_chunked_files.py

This script combines multiple CSV files present in a specified directory into a single CSV file.
It reads the header row from the first input file and writes it to the output file, followed by
all remaining rows from each input file.

Usage:
    python combine_chunked_files.py --input /path/to/input/directory

The script expects the following command-line argument:
    -i, --input: Path to the directory containing the input CSV files

The combined CSV file will be written to a subdirectory named "combined_reviews" within the
specified input directory, with the filename "combined_reviews.csv".

Functions:
    combine_csv_files(input_files, output_file):
        Combines multiple CSV files into a single output file.
        input_files: List of paths to the input CSV files.
        output_file: Path to the output CSV file.

    get_csv_filenames(directory):
        Returns a list of CSV filenames present in the specified directory.
        directory: Path to the directory.

    parse_arguments():
        Parses the command-line arguments and returns the input directory path.

Example:
    python combine_chunked_files.py --input /path/to/input/directory
"""

import os
import csv
import argparse


def combine_csv_files(input_files, output_file):
    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)

        # read header from first file
        with open(input_files[0], "r") as first_file:
            reader = csv.reader(first_file)
            headers = next(reader)
            writer.writerow(headers)

        # write all rows from files to output file
        for filename in input_files:
            with open(filename, "r") as infile:
                reader = csv.reader(infile)
                next(reader)
                for row in reader:
                    writer.writerow(row)


def get_csv_filenames(directory):
    filenames = []
    for filename in os.listdir(directory):
        if "csv" in filename:
            filenames.append(filename)
    return filenames


def parse_arguments():
    parser = argparse.ArgumentParser(description="Directory path")
    parser.add_argument("-i", "--input", help="Path to the input files")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # take path of the chunked files as command line input.
    filepath = parse_arguments().input

    # Fetch csv filenames present in the directory
    filenames = get_csv_filenames(filepath)

    # create file paths.
    input_file_paths = [os.path.join(filepath, file) for file in filenames]

    # combine files
    output_file = os.path.join(
        filepath,
        "combined_reviews",
        "combined_reviews.csv",
    )

    # combine chunked files into one and write to output directory
    combine_csv_files(input_file_paths, output_file=output_file)
    # print(get_csv_filenames(processed_reviews))
