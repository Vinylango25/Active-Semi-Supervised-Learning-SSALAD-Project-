import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.preprocessing import StandardScaler
import logging

from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.iforest import IForest


import xgboost as xgb

from utils.label_query import query_labels

import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(filename='experiment_errors.log', level=logging.ERROR, format='%(asctime)s %(message)s')

# define the models
PYOD_SELECTED_MODELS = {
    'PCA': PCA,
    'KNN': KNN,
    'IForest': IForest,
}



def train_xgboost_regressor(X_train, y_train, X_test, y_test, random_seed, propagation):
    if propagation and y_train.shape[1]>1:
        xgb_model = xgb.XGBRegressor(random_state=random_seed)
        mask = ~np.isnan(y_train[:,0])
        xgb_model.fit(X_train[mask,:], -y_train[mask,0])
        xgb_pred = xgb_model.predict(X_test)
    else:
        xgb_model = xgb.XGBClassifier(random_state=random_seed)
        # only get values that are 0 or 1
        mask = y_train > -1
        xgb_model.fit(X_train[mask,:], y_train[mask])
        xgb_pred = xgb_model.predict_proba(X_test)[:,1]
    try:
        roc_auc = roc_auc_score(y_test, xgb_pred)
        return roc_auc, xgb_model
    except Exception as e:
        print(f"Exceptions raise at XGBOOST:  {e}")
        return None, xgb_model
        
# function to prepare data
def get_sampled_data(X, y, samples_frac, random_state=42):
    _, Xsamp, _, ysamp = train_test_split(X, y, test_size=samples_frac,
                                        random_state=random_state,
                                        stratify=y)
    return Xsamp, ysamp

# ftn to perform label propagation if needed
def label_propagation(X_train, labels, random_state=42, kernel='knn'):
    label_prop_model = LabelPropagation(kernel=kernel)
    label_prop_model.fit(X_train, labels)
    label_proba = label_prop_model.predict_proba(X_train)

    # replace label_proba with labels where labels are 0 or 1
    mask = labels > -1
    label_proba[mask,0] = 1 - labels[mask]

    
    rng = np.random.RandomState(random_state)
    random_values = rng.rand(len(X_train))
    
    # remove outliers based on the random values
    remove_idx = np.where(label_proba[:, 0] < random_values)[0]
    X_train = np.delete(X_train, remove_idx, axis=0)
    labels = np.delete(labels, remove_idx)
    
    return X_train, labels, label_proba


# functiom to evaluate model and log results
def train_and_evaluate_model(X_train, X_test, y_train, y_test, semi_detector):
    try:
        # train model
        semi_detector.fit(X_train)
        X_test_scores = semi_detector.decision_function(X_test)
        # find ROC-AUC scores
        roc_auc = roc_auc_score(y_test, X_test_scores)
    except ValueError as e:
        print(f"Error occured while getting the auc values : e={e}")
        # handle error cases where AUC cannot be calculated E.G single class in the test set
        roc_auc = np.nan
        
    return roc_auc, semi_detector

# main experiment function
def run_experiment(X, y, num_samples=1000, reps=5, fractions=None, model_names=None, query_strategies=None, propagations=None, dataset_name="", kernel='knn'):
    """
        Run an experiment for active learning and semi-supervised learning on a given dataset, evaluating different query strategies, models, and configurations.
    
        The ftn trains specified set of models using different query strategies and evaluates their performance with a combination of label propagation and query-based sampling.
    
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix where each row represents a sample and each column a feature.
    
        y : array-like, shape (n_samples,)
            True labels for the samples.
    
        num_samples : int, optional, default=1000
            The number of samples to use for training in each repetition.
    
        reps : int, optional, default=5
            The number of repetitions to run the experiment. Results will be averaged over these repetitions. The value is also used as a randon state value
    
        fractions : list of float, optional, default=None
            The fractions of labeled data to query in each iteration. If None, a default set of fractions will be used.
    
        model_names : list of str, optional, default=None
            List of model names to use for the experiment (e.g., ['IForest']). If None, a default model will be used.
    
        query_strategies : list of str, optional, default=None
            List of query strategies to evaluate. Options are: 
            - 'random', 'uncertainty', 'anomalous', 'clusters'. If None, all strategies are used.
    
        propagations : list of bool, optional, default=None
            List of boolean values indicating whether label propagation should be applied after querying labels.
    
        dataset_name : str, optional, default=""
            Name of the dataset, which will be recorded in the experiment results.
    
        Returns:
        --------
        all_results : pandas DataFrame
            DataFrame containing the results of the experiment
    
        Notes:
        ------
        - The function uses active learning techniques by querying a fraction of the training data based on different strategies.
        - It evaluates the models by computing the ROC AUC score on the test set.
        - Results are stored and returned in a pandas DataFrame for easy analysis.
    """

    if fractions is None:
        fractions = [0, 0.004, 0.006, 0.008, 0.01, 0.013, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0]
    if model_names is None:
        model_names = ['IForest']
    if query_strategies is None:
        query_strategies = ['random', 'uncertainty', 'anomalous', 'clusters']
    if propagations is None:
        propagations = [True, False]
    all_results = pd.DataFrame()


    # scaler the data
    scaler = StandardScaler()
    X_ = X.copy()
    X = scaler.fit_transform(X)


       # get samples from dataset
    samples_frac = np.minimum(num_samples / np.shape(X)[0], 1.0)

    for frac in tqdm(fractions):
        test_size = 0.5
        print(f"\nDataset: {dataset_name}, Fraction: {frac}, Test size: {test_size}")
        if not (0.0 < test_size < 1.0):
            error_message = f"Invalid test_size: {test_size} for fraction: {frac} in dataset: {dataset_name}"
            print(error_message)
            logging.error(error_message)
            continue

        for model_name in model_names:
            # define the semi supervised model to be used
            semi_detector_clf = PYOD_SELECTED_MODELS[model_name]()
            for strategy in query_strategies:
                for propagation in propagations:
                    for rep in range(reps):
                        # step1: data sample for training
                        if samples_frac <1:
                            Xsamp, ysamp = get_sampled_data(X, y, samples_frac, random_state=rep)
                        else:
                            Xsamp, ysamp = X.copy(), y.copy()

                        if len(np.unique(ysamp)) < 2:  # If there are fewer than two classes
                            print(f"Warning: Dataset has fewer than 2 classes for replication {rep}. Skipping.")
                            continue

                        # step2: Split into training and testing sets
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(Xsamp, ysamp, test_size=test_size, random_state=rep, stratify=ysamp)
                        except ValueError as e:
                            error_message = f"Error during train_test_split for dataset: {dataset_name}, fraction: {frac}, rep: {rep} - {e}"
                            print(error_message)
                            logging.error(error_message)
                            continue

                        #unsupervised model, this is used when strategy is uncertainity/anomalous
                        unsup_detector = PYOD_SELECTED_MODELS[model_name]()
                        unsup_detector.fit(X_train)


                        # step3: query labels if needed
                        num_labels = int(frac * len(y_train))
                        if num_labels > 0:
                            labels = query_labels(X_train, y_train, detector=unsup_detector, strategy=strategy, num_labels=num_labels)
                            
                            # perfom propagation if enabled
                            if propagation:
                                X_train_unsup, y_train_unsup, label_proba = label_propagation(X_train, labels, random_state=rep, kernel=kernel)
                            else:
                                remove_idx = np.where(labels == 1)[0]
                                X_train_unsup = np.delete(X_train, remove_idx, axis=0)
                                y_train_unsup = np.delete(y_train, remove_idx)
                        else:
                            X_train_unsup = X_train.copy()
                            y_train_unsup = y_train.copy()

                        # step4, train and get the metrics on test data
                        current_result = {
                            'dataset': dataset_name, 
                            'num_samples': len(ysamp),
                            'model': model_name, 
                            'fraction': frac, 
                            'rep': rep,
                            'query_strategy': strategy, 
                            'propagation': propagation,
                            'kernel': kernel,
                            'num_labels': num_labels
                        }
                        if not np.shape(X_train_unsup)[0] > 0:  # in case where label prop tells us to remove all data
                            X_train_unsup = X_train.copy()
                            y_train_unsup = y_train.copy()
                        curr_auc_score, semi_detector_clf = train_and_evaluate_model(
                            X_train_unsup.copy(), 
                            X_test, 
                            y_train_unsup, 
                            y_test, 
                            semi_detector_clf
                        )
                        current_result["roc_auc"] = curr_auc_score
                        result_df = pd.DataFrame(current_result, index=[0])
                        all_results = pd.concat([all_results, result_df], ignore_index=True)

                        #perfoming xgboost call
                        if num_labels > 0 and 0 in labels and 1 in labels:
                            #use the original Xtrain data and its probalities incase we did propagation else just the actual labels queried
                            reg_auc_score, regressor = train_xgboost_regressor(X_train, label_proba if propagation else labels, X_test, y_test, random_seed=rep, propagation=propagation)
                            if reg_auc_score:
                                current_result["roc_auc"] = reg_auc_score
                                current_result['model'] = "XGB"
                                result_df = pd.DataFrame(current_result, index=[0])
                                all_results = pd.concat([all_results, result_df], ignore_index=True)

    return all_results
