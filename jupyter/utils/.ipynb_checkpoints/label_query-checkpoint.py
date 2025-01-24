from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import numpy as np


def query_labels(X, y, detector,
                 queried_labels=None, strategy='random',
                 num_labels=1, random_state=42):


    """
        query labels from a dataset X using different active learning strategies.
    
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix where each row represents a sample and each column a feature.
        
        y : array-like, shape (n_samples,)
            True labels for the samples.
        
        detector : object
            A model or detector used to estimate uncertainty or outlier scores (must have object.predict_confidence and object.decision_function methods)
        
        queried_labels : array-like, shape (n_samples,), optional, default=None
            Array indicating which labels have already been queried. Unqueried labels should be marked with -1. 
            If None, all labels are initialized as unqueried.
        
        strategy : {'random', 'uncertainty', 'anomalous', 'clusters'}, optional, default='random'
            The strategy used to select which labels to query. Options:
            - 'random': Randomly select unqueried labels.
            - 'uncertainty': Select labels based on the uncertainty of the detector's predictions.
            - 'anomalous': Select labels based on the highest outlier scores from the detector.
            - 'clusters': Cluster the unqueried samples and select the medoid of each cluster.
        
        num_labels : int, optional, default=1
            The number of labels to query in the current batch.
    
        random_state : int, RandomState instance, or None, optional, default=42
            Seed for the random number generator, or None for random initialization.
        
        Returns:
        --------
        queried_labels : array-like, shape (n_samples,)
            The updated array of labels where -1 indicates unqueried samples and the queried labels are filled in.
    """
    
    if queried_labels is None:  # initialize all labels as unqueried (-1)
        queried_labels = np.zeros(len(y)) - 1

    idx_unqueried = np.where(queried_labels == -1)[0]
    
    if strategy == 'random':
        rng = np.random.default_rng(random_state)
        idx_random = rng.permuted(idx_unqueried)
        queried_labels[idx_random[:num_labels]] = y[idx_random[:num_labels]]

    elif strategy == 'uncertainty':
        # get outlier confidence
        confidence = detector.predict_confidence(X[idx_unqueried,:])
        # sort by confidence
        idx_confidence = np.argsort(confidence)
        queried_labels[idx_confidence[:num_labels]] = y[idx_confidence[:num_labels]]

    elif strategy == 'anomalous':
        # get outlier scores
        scores = detector.decision_function(X[idx_unqueried,:])
        # sort by decreasing outlier score
        idx_scores = np.argsort(scores)[::-1]
        queried_labels[idx_scores[:num_labels]] = y[idx_scores[:num_labels]]

    elif strategy == 'clusters':
        # get clusters
        clust = AgglomerativeClustering(n_clusters=num_labels)
        clust_labels = clust.fit_predict(X[idx_unqueried,:])
        # for each cluster, get the medoid
        for i in range(num_labels):
            idx_cluster = np.where(clust_labels == i)[0]
            dist_mat = pairwise_distances(X[idx_cluster,:])
            medoid_index = np.argmin(dist_mat.sum(axis=1))
            queried_labels[idx_cluster[medoid_index]] = y[idx_cluster[medoid_index]]

    return queried_labels