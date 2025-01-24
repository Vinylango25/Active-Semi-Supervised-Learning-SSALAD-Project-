import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.sod import SOD
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.kde import KDE
from pyod.models.cblof import CBLOF


from utils.label_query import query_labels

import warnings
warnings.filterwarnings("ignore")



# define the models
PYOD_SELECTED_MODELS = {
    'PCA': PCA,
    'OCSVM': OCSVM,
    'LOF': LOF,
    'COF': COF,
    'HBOS': HBOS,
    'KNN': KNN,
    'SOD': SOD,
    'COPOD': COPOD,
    'ECOD': ECOD,
    'IForest': IForest,
    'LODA': LODA,
    'KDE': KDE,
    'CBLOF': CBLOF
}


# function to prepare data
def get_sampled_data(X, y, num_samples, random_state=42):
    samp_frac = num_samples / np.shape(X)[0]
    _, Xsamp, _, ysamp = train_test_split(X, y, test_size=samp_frac,
                                        random_state=random_state,
                                        stratify=y)
    return Xsamp, ysamp

# ftn to perform label propagation if needed
def label_propagation(X_train, labels, random_state=42):
    label_prop_model = LabelPropagation()
    label_prop_model.fit(X_train, labels)
    label_proba = label_prop_model.predict_proba(X_train)
    
    rng = np.random.RandomState(random_state)
    random_values = rng.rand(len(X_train))
    
    # remove outliers based on the random values
    remove_idx = np.where(label_proba[:, 0] < random_values)[0]
    X_train = np.delete(X_train, remove_idx, axis=0)
    labels = np.delete(labels, remove_idx)
    
    return X_train, labels


# functiom to evaluate model and log results
def train_and_evaluate_model(X_train, X_test, y_train, y_test, semi_detector):
    # train model
    semi_detector.fit(X_train)
    X_test_scores = semi_detector.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, X_test_scores)

    return roc_auc, semi_detector

# main experiment function
def run_experiment(X, y, num_samples=1000, reps=5, fractions=None, model_names=None, query_strategies=None, propagations=None, dataset_name=""):
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


    for frac in tqdm(fractions):
        for model_name in model_names:
            # define the semi supervised model to be used
            semi_detector_clf = PYOD_SELECTED_MODELS[model_name]()
            for strategy in query_strategies:
                for propagation in propagations:
                    for rep in range(reps):
                        # step1: data sample for training
                        Xsamp, ysamp = get_sampled_data(X, y, num_samples, random_state=rep)

                        # step2: Split into training and testing sets
                        X_train, X_test, y_train, y_test = train_test_split(Xsamp, ysamp, test_size=0.5, random_state=rep, stratify=ysamp)

                        #unsupervised model, this is used when strategy is uncertainity/anomalous
                        unsup_detector = PYOD_SELECTED_MODELS[model_name]()
                        unsup_detector.fit(X_train)


                        # step3: query labels if needed
                        num_labels = int(frac * len(y_train))
                        if num_labels > 0:
                            labels = query_labels(X_train, y_train, detector=unsup_detector, strategy=strategy, num_labels=num_labels)
                            
                            # perfom propagation if enabled
                            if propagation:
                                X_train, y_train = label_propagation(X_train, labels, random_state=rep)
                            else:
                                remove_idx = np.where(labels == 1)[0]
                                X_train = np.delete(X_train, remove_idx, axis=0)
                                y_train = np.delete(y_train, remove_idx)

                        # step4, train and get the metrics on test data
                        current_result = {
                            'dataset': dataset_name, 
                            'num_samples': len(y_train),
                            'model': model_name, 
                            'fraction': frac, 
                            'rep': rep,
                            'query_strategy': strategy, 
                            'propagation': propagation
                        }
                        
                        curr_auc_score, semi_detector_clf = train_and_evaluate_model(
                            X_train, 
                            X_test, 
                            y_train, 
                            y_test, 
                            semi_detector_clf
                        )
                        current_result["roc_auc"] = curr_auc_score
                        result_df = pd.DataFrame(current_result, index=[0])

                        # step5: merge results
                        all_results = pd.concat([all_results, result_df], ignore_index=True)

    return all_results
