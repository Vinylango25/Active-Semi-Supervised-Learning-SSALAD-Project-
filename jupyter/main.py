import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm.contrib.concurrent import thread_map
import threading

from utils.label_query import query_labels
from utils.result_plotter import plot_ssalad_results
from utils.semi_supervised_trainer import run_experiment

threads = 3

curr_fractions = [0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.013, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0]
all_models = ['IForest']
all_datasets = ['11_donors', '33_skin', '26_optdigits', '45_wine', '12_fault', '30_satellite', '37_Stamps']
# Verify the presence of dataset files
data_dir = './data'
available_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]

if 'ALL' in all_datasets:
    all_datasets = [f.split('.npz')[0] for f in available_files]

propagation_kernels = ["knn"]
RESULTS_FOLDER = "results_26_optdigits"

total_datasets = len(all_datasets)

all_kernel_results = {
    # "rbf": pd.DataFrame(),
    "knn": pd.DataFrame()
}

# a folder to add results if it doesn't exist
os.makedirs(f'{RESULTS_FOLDER}', exist_ok=True)
for each_kernel in propagation_kernels:
    # create folder for kernels and their results
    os.makedirs(f"{RESULTS_FOLDER}/{each_kernel}", exist_ok=True)
    # to store individual data csvs: Helps to recover in case of errors
    os.makedirs(f"{RESULTS_FOLDER}/{each_kernel}/csv", exist_ok=True)
    # for png files
    os.makedirs(f"{RESULTS_FOLDER}/{each_kernel}/png", exist_ok=True)

# Create a lock for thread-safe printing
print_lock = threading.Lock()

def process_dataset(args):
    selected_dataset, kernel_name = args
    path = f'{data_dir}/{selected_dataset}.npz'
    if not os.path.exists(path):
        with print_lock:
            print(f"File not found: {path}")
        return None
    data = np.load(path, allow_pickle=True)
    X, y = data['X'], data['y']
    num_samples = 2000
    results = run_experiment(
        X,
        y,
        num_samples=num_samples,
        model_names=all_models,
        fractions=curr_fractions,
        dataset_name=selected_dataset,
        kernel=kernel_name
    )
    results.to_csv(f"{RESULTS_FOLDER}/{kernel_name}/csv/{selected_dataset}_run.csv", index=False)
    return results, selected_dataset, kernel_name

# Use thread_map to display the progress bar properly
results = thread_map(process_dataset, [(dataset, kernel) for kernel in propagation_kernels for dataset in all_datasets], max_workers=threads, desc="Processing datasets")

for result, dataset, kernel in results:
    if result is not None:
        if kernel not in all_kernel_results:
            all_kernel_results[kernel] = pd.DataFrame()
        all_kernel_results[kernel] = pd.concat([all_kernel_results[kernel], result])
    with print_lock:
        print(f"Completed: {dataset} with kernel {kernel}")

# save all results
for kernel in propagation_kernels:
    all_kernel_results[kernel].to_csv(f"{RESULTS_FOLDER}/{kernel}/all_run.csv", index=False)

# plot
for each_kernel in propagation_kernels:
    kernel_df = all_kernel_results[each_kernel]
    for model_num, each_model in enumerate(kernel_df["model"].unique()):
        curr_results = kernel_df[(kernel_df['model'] == each_model)]
        if curr_results.shape[0] <= 0:
            with print_lock:
                print(f"No results for model = {all_models[0]} and data = {each_data}")
            continue
        plot_ssalad_results(curr_results, log_scale=True, results_path="results_26_optdigits")

        