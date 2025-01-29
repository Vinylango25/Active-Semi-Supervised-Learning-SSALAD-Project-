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

def process_dataset(selected_dataset):
    path = f'{data_dir}/{selected_dataset}.npz'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    data = np.load(path, allow_pickle=True)
    X, y = data['X'], data['y']
    num_samples = 1000
    if X.shape[0] <= num_samples:
        num_samples = int(X.shape[0] // 2)
    results = run_experiment(X, y, num_samples=num_samples, model_names=all_models, fractions=curr_fractions, dataset_name=selected_dataset)
    results_df = pd.concat([results_df, results])
    results.to_csv(f"{selected_dataset}_run.csv", index=False)
    return results

results_df = pd.DataFrame()
total_datasets = len(all_datasets)

with ThreadPoolExecutor(max_workers=16) as executor:
    future_to_dataset = {executor.submit(process_dataset, dataset): dataset for dataset in all_datasets}
    for idx, future in enumerate(as_completed(future_to_dataset)):
        dataset = future_to_dataset[future]
        try:
            result = future.result()
            if result is not None:
                results_df = pd.concat([results_df, result])
            print(f"Completed {idx + 1}/{total_datasets}: {dataset}")
        except Exception as exc:
            print(f"{dataset} generated an exception: {exc}")

# save the results
results_df.to_csv(f"all_run.csv", index=False)

# loop through datasets
for each_data in all_datasets:
    print(f"Plotting results for dataset: {each_data}")
    curr_results_queries = results_df[results_df['dataset'] == each_data]
    if curr_results_queries.shape[0] <= 0:
        print(f"No results for model = {all_models[0]} and data = {each_data}")
        continue
    plot_ssalad_results(curr_results_queries, model_name=all_models[0], dataset=each_data, log_scale=True)
    plt.savefig(f"{all_models[0]}_for_{each_data}_fig.png")
    plt.tight_layout()
    plt.show()


