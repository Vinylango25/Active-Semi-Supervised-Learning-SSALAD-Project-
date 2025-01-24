import numpy as np
import pandas as pd
from utils.label_query import query_labels
from utils.result_plotter import plot_ssalad_results
from utils.semi_supervised_trainer import run_experiment

dataset = '33_skin'
path = f'data/{dataset}.npz'
data = np.load(path, allow_pickle=True)
X, y = data['X'], data['y']

curr_fractions = [0, 0.004, 0.006, 0.008, 0.013, 0.02, 0.03, 0.04, 0.06, 0.08,
             0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0]

results = run_experiment(X, y, fractions=curr_fractions, dataset_name=dataset)


# save the results
results.to_csv(f"{dataset}_run.csv", index=False)
#  a plot
plt.figure(figsize=(15, 6))
plot_ssalad_results(results, "IFOrest", "SKIN", log_scale=True, ax=plt.gca())
plt.tight_layout()
plt.savefig(f"{dataset}_fig.png")
plt.show()


