import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.label_query import query_labels
from utils.result_plotter import plot_ssalad_results
from utils.semi_supervised_trainer import run_experiment


curr_fractions = [0, 0.004, 0.006, 0.008, 0.013, 0.02, 0.03, 0.04, 0.06, 0.08,
             0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0]
all_datasets = ['33_skin', '11_donors']#, '2_annthyroid', '6_cardio', '7_Cardiotocography']
all_models =  ['PCA', 'KNN', 'IForest']#, 'OCSVM', 'LOF', 'COF', 'HBOS', 'COPOD', 'ECOD', 'LODA', 'KDE', 'CBLOF']



results_df = pd.DataFrame()
for selected_datset in all_datasets:
    path = f'data/{selected_datset}.npz'
    data = np.load(path, allow_pickle=True)
    X, y = data['X'], data['y']
    results = run_experiment(X, y, fractions=curr_fractions, dataset_name=selected_datset)
    results_df = pd.concat([results_df, results])
    results.to_csv(f"{selected_datset}_run.csv", index=False)

# save the results
results_df.to_csv(f"all_run.csv", index=False)

#  a plot
# loop through datasets
for each_data in all_datasets:
    for model_num, each_model in enumerate(all_models):
        curr_results_queries = results[(results['dataset'] == each_data) & (results['dataset'] == each_data)]
        if curr_results_queries.shape[0]<=0:
            print(f"No results for model = {each_model} and data = {each_data}")
            continue
            
        plot_ssalad_results(curr_results_queries, each_model, each_data, log_scale=True)
        plt.savefig(f"{each_model}_for_{each_data}_fig.png")
        plt.tight_layout() 
        plt.show()


