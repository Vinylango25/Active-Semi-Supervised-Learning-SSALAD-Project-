import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.label_query import query_labels
from utils.result_plotter import plot_ssalad_results
from utils.semi_supervised_trainer import run_experiment

curr_fractions = [0, 0.004, 0.006, 0.008, 0.01, 0.013, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0]
all_models =  ['IForest']
#all_datasets = ['33_skin', '2_annthyroid', '11_donors', '6_cardio', '7_Cardiotocography']
all_datasets = ['14_glass','15_Hepatitis','21_Lymphography','29_Pima','37_Stamps','39_vertebral','4_breastw','43_WDBC','45_wine','46_WPBC']
#all_datasets = ['ALL']

# Verify the presence of dataset files
data_dir = './data'
available_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
print(f"Available files in {data_dir}: {available_files}")

# If 'ALL' is specified, use all datasets in the ../data directory
if 'ALL' in all_datasets:
    all_datasets = [f.split('.npz')[0] for f in available_files]

# all datset: ADBENCH:---> (one plot AUC score to unsupervised...)


propagation_kernels = ["rbf", "knn"]
RESULTS_FOLDER = "results"

total_datasets = len(all_datasets)

all_kernel_results = {
    "rbf": pd.DataFrame(),
    "knn": pd.DataFrame()
}


# a folder to add results if it doesn't exist
os.makedirs(f'{RESULTS_FOLDER}', exist_ok=True)
for each_kernel in propagation_kernels:
    #create folder for kernels and their results
    os.makedirs(f"{RESULTS_FOLDER}/{each_kernel}", exist_ok=True)
    # to store individual data csvs: Helps to recover incase of errors
    os.makedirs(f"{RESULTS_FOLDER}/{each_kernel}/csv", exist_ok=True)
    #for png files
    os.makedirs(f"{RESULTS_FOLDER}/{each_kernel}/png", exist_ok=True)


def process_dataset(selected_dataset, kernel_name):
    path = f'{data_dir}/{selected_dataset}.npz'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    data = np.load(path, allow_pickle=True)
    X, y = data['X'], data['y']
    num_samples = 1000
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
    return results



with ThreadPoolExecutor(max_workers=4) as executor:
    # future_to_dataset = {executor.submit(process_dataset, dataset, each_kernel): dataset for dataset in all_datasets}
    future_to_dataset = {executor.submit(process_dataset, dataset, kernel): (dataset, kernel) 
                         for kernel in propagation_kernels for dataset in all_datasets}
    for idx, future in enumerate(as_completed(future_to_dataset)):
        dataset, kernel = future_to_dataset[future]
        try:
            result = future.result()  
            if result is not None:
                if kernel not in all_kernel_results:
                    all_kernel_results[kernel] = pd.DataFrame()
                all_kernel_results[kernel] = pd.concat([all_kernel_results[kernel], result])
            print(f"Completed {idx + 1}/{len(all_datasets) * len(propagation_kernels)}: {dataset} with kernel {kernel}")
        except Exception as exc:
            print(f"{dataset} with kernel {kernel} generated an exception: {exc}")
    
    #save all results
    for kernel in propagation_kernels:
        all_kernel_results[kernel].to_csv(f"{RESULTS_FOLDER}/{kernel}/all_run.csv", index=False)

##plot
for each_kernel in propagation_kernels:
    kernel_df = all_kernel_results[each_kernel]
    for each_data in all_datasets:
        curr_results = kernel_df[kernel_df['dataset'] == each_data]
        if curr_results.shape[0] <= 0:
            print(f"No results for model = {all_models[0]} and data = {each_data}")
            continue
        plot_ssalad_results(curr_results, model_name=all_models[0], dataset=each_data, log_scale=True)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_FOLDER}/{each_kernel}/png/{each_data}.png")

        # plt.show()

