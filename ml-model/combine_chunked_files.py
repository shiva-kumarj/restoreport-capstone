"""combine processed chunk files into one"""

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
