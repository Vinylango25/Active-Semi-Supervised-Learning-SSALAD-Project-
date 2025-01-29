import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.label_query import query_labels
from utils.result_plotter import plot_ssalad_results
from utils.semi_supervised_trainer import run_experiment


curr_fractions = [0, 0.004, 0.006, 0.008, 0.01, 0.013, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0]
all_datasets = ['1_ALOI', '2_annthyroid', '3_backdoor', '4_breastw', '5_campaign', '6_cardio', '7_Cardiotocography', 
            '8_celeba', '9_census', '10_cover', '11_donors', '12_fault', '13_fraud', '14_glass', '15_Hepatitis',
            '16_http', '17_InternetAds', '18_Ionosphere', '19_landsat', '20_letter', '21_Lymphography', 
            '22_magic.gamma', '23_mammography', '24_mnist', '25_musk', '26_optdigits', '27_PageBlocks',
            '28_pendigits', '29_Pima', '30_satellite', '31_satimage-2', '32_shuttle', '33_skin', 
            '34_smtp', '35_SpamBase', '36_speech', '37_Stamps', '38_thyroid', '39_vertebral', 
            '40_vowels', '41_Waveform', '42_WBC', '43_WDBC', '44_Wilt', '45_wine', '46_WPBC', '47_yeast']

all_models =  ['KNN', 'IForest']

# all datset: ADBENCH:---> (one plot AUC score to unsupervised...)

results_df = pd.DataFrame()
for selected_datset in all_datasets:
    path = f'data/{selected_datset}.npz'
    data = np.load(path, allow_pickle=True)
    X, y = data['X'], data['y']
    num_samples = 1000
    results = run_experiment(X, y, num_samples=num_samples, model_names = all_models, fractions=curr_fractions, dataset_name=selected_datset)
    results_df = pd.concat([results_df, results])
    results.to_csv(f"{selected_datset}_run.csv", index=False)

# save the results
results_df.to_csv(f"all_run.csv", index=False)

# loop through datasets
for each_data in all_datasets:
    for model_num, each_model in enumerate(all_models):
        curr_results_queries = results_df[(results_df['dataset'] == each_data) & (results_df['dataset'] == each_data)]
        if curr_results_queries.shape[0]<=0:
            print(f"No results for model = {each_model} and data = {each_data}")
            continue
        plot_ssalad_results(curr_results_queries, model_name=each_model, dataset=each_data, log_scale=True)
        plt.savefig(f"{each_model}_for_{each_data}_fig.png")
        plt.tight_layout() 
        plt.show()


