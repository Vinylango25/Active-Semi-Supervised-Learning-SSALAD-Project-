# Semi-Supervised Learning Experiment

The repo contains a running guide for SSALAD experiment on multiple datasets using different models. The experiments output results in terms of AUC scores, and the results are saved to CSV files for further analysis. 

In order to run the experiment, you need to ensure that the necessary datasets and Python packages are installed and set up. Follow the instructions below. 

### 1. Download the Datasets
First You need to move to `jupyter` folder i.e
```bash
cd jupyter
```


If `data` folder is not available (there are no dataset) for running the experiment, you need to download the required datasets by running  `data_downloader.sh` script i.e.

```bash
./data_downloader.sh
```

This will download the datasets into the `data/` directory.

### 2. Install Required Packages

Have Python 3.x installed, then install the required Python packages. You can install the necessary dependencies using `pip`:

```bash
pip install pyod scikit-learn matplotlib seaborn
```

These packages are used for:
- **PyOD**: A Python library for detecting outliers (used for semi-supervised learning models).
- 

### 3. Running the Experiment

when all req. are met, you can run the main experiment script, `main.py`.

To run the experiment, execute the following:

```bash
python main.py
```

This will load the datasets, run the experiment with the specified models, and output the results on `all_run.csv`.

### 4. Experiment Customization

You can modify which datasets and models are used by editing the `all_datasets` and `all_models` variables in the script.

#### e.g:

```python
all_datasets = ['33_skin']
all_models = ['IForest']
```

- **`all_datasets`**: List of datasets to run the experiment on (datasets must be in the `data/` directory).
- **`all_models`**: List of models to evaluate.
- 


### 6. Files Generated
- On the folder defined in main i.e `results` graphs and csv files will be generated.
- `all_run.csv`: A CSV file containing the results of all experiments,
- `{dataset}_run.csv`: A CSV file containing the results for a specific dataset.
- `{model}_for_{dataset}_fig.png`: A plot showing the AUC scores for a specific model and dataset.

### 7. Code Overview

- **`utils/label_query.py`**:  Functions for querying labels in the dataset.
- **`utils/result_plotter.py`**: Funtion for plotting the results of the experiment.
- **`utils/semi_supervised_trainer.py`**: Function which runs the semi-supervised learning experiment for each dataset and model.


### Reseach Papers References:
- For some ideas on the working, some papers were reviewed which ca be seen on the folder `./jupyter/reseach_papers`. It has the following papers;
    - SPADE: Semi-supervised Anomaly Detection under Distribution Mismatch
    - Label Propagation for Deep SSL (generates a nearest neighbor graph and conducts label propagation through transductive learning on the training set.)
    - ALIF (an extension of the already popular Isolation Forest (IF) anomaly detection model).


