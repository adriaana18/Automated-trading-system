import concurrent.futures
import logging
import threading
import time
import os


import LSTM_price_predictions_threads as LSTM


csv_dir = 'C:/Users/adria/OneDrive/Documents/RAE/Code'

# Crate a function that accesses the csv files direcly.
def get_csf_files(files_dir):
    all_files = []
    for file in os.listdir(files_dir):
        if file.endswith(".csv"):
            all_files.append(file)
    return all_files

# Create a function that pulls multiple stocks and trains the LSTM at once for all of them.
def thread_function(item):
    logging.info("Thread %s: starting", item)
    LSTM.calculate_predictions(item)
    logging.info("Thread %s: finishing", item)

if __name__ == "__main__":
    max_threads = 15
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        executor.map(thread_function, get_csf_files(csv_dir))




