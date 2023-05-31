import pandas as pd

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split


def get_cluster_scores(df, labels):
    sil = silhouette_score(df, labels)
    cali = calinski_harabasz_score(df, labels)
    davi = davies_bouldin_score(df, labels)
    score_record = {
        "silhouette_score": sil,
        "calinski_harabasz_score": cali,
        "davies_bouldin_score": davi
    }
    print(f"Silhouette score: {sil} \n Calinski Harabasz score: {cali} \n Davies Bouldin Score: {davi}")
    return score_record
    

def get_cluster_distribution(labels: list):
    labels = pd.Series(labels)
    dist = labels.value_counts(normalize=True)
    print("+++ Cluster Percentages: +++")
    for cluster, p in dist.items():
        percentage = p * 100
        print(f"Cluster {cluster}: {percentage:.2f}%")
    return dist


# @TODO: get lat long for map demo
def get_random_samples(df, labels, size=5):
    result_df = pd.DataFrame(labels, columns=['cluster'], index=df.index)
    unique_clusters = result_df.cluster.unique()
    samples = {}
    for cluster in unique_clusters:
        temp_df = result_df[result_df['cluster'] == cluster]
        sample_stores = temp_df.sample(n=size, replace=False).index.tolist()
        samples[cluster] = sample_stores
        print("Random samples for cluster", cluster)
        print([f'Store {i}' for i in sample_stores], "\n")
    return samples


def get_top_features_by_variance(df, labels):
    """
    Get the top features based on homogeneity within the cluster.

    :param df: Original dataframe passed to the model
    :param labels: Model clustering results (same dimension as input size)
    :return: Result dict with clustering result label as key and a list of top features
    """
    df['label'] = labels
    label_groups = df.groupby(by="label")
    label_features = {}
    for label, group in label_groups:
        group_std = group.std()
        group_min = group.min()
        group_max = group.max()
        group_metric = 1 - (group_std / (group_max - group_min))
        group_metric = group_metric.sort_values(ascending=False)
        label_features[label] = group_metric[:10]
    return label_features



