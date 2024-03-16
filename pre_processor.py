import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class PreProcessor:
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.temp = list()

    def set_index(self, index_columns):
        self.data_frame.set_index(index_columns, inplace=True)

    def drop_columns(self, columns):
        self.data_frame.drop(columns=columns, inplace=True)

    def drop_na(self, how, axis):
        self.data_frame.dropna(how=how, axis=axis, inplace=True)

    def fill_na(self, value, method, axis=0):
        self.data_frame.fillna(value=value, method=method, axis=axis, inplace=True)

    def drop_duplicates(self, subset, keep):
        self.data_frame.drop_duplicates(subset=subset, keep=keep, inplace=True)

    def replace(self, value, method, regex):
        self.data_frame.replace(value, method=method, regex=regex, inplace=True)

    def merge(self, other, left_on, right_on,how):
        self.data_frame = pd.merge(left=self.data_frame, right=other, left_on=left_on, right_on=right_on, how=how)

    def group_by(self, group_column, columns_agg):
        return self.data_frame.groupby(group_column).agg(columns_agg)

    def rename(self, columns):
        self.data_frame = self.data_frame.rename(columns=columns, inplace=True)

    def normalize(self, columns=None):
        if columns is not None:
            data = self.data_frame.loc[:, columns]
            data = (data - data.mean(axis=0)) / data.std(axis=0)
            self.data_frame.loc[:, columns] = data
        else:
            columns = self.data_frame.select_dtypes(include='number').columns
            data = self.data_frame.loc[:, columns]
            data = (data - data.mean(axis=0)) / data.std(axis=0)
            self.data_frame.loc[:, columns] = data

    def one_hot_encoder(self, columns=None):
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        if columns is None:
            columns = self.data_frame.select_dtypes(include='object').columns
        encoder.fit(self.data_frame.loc[:, columns])
        one_hot_encoded = encoder.transform(self.data_frame.loc[:, columns])
        encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out())
        encoded_df.index = self.data_frame.index
        self.data_frame = pd.concat([self.data_frame.drop(columns, axis=1), encoded_df], axis=1)


class FeatureSelection:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def add_log_transform(self, columns):
        for col in columns:
            self.data_frame[f'log_{col}'] = np.log1p(self.data_frame[col])

    def extend_data_by_hdbscan(self, columns, eps=0.5, min_samples=5, metric='euclidean', p=None):
        data = self.data_frame.loc[:, columns]
        self.data_frame['hdbscan_cluster'] = HDBSCAN(min_samples=min_samples, metric=metric).fit(data).labels_

    def reduction_dimension_by_pca(self, scaled_columns, handel='return', index=''):
        # Reduce dimensionality using PCA
        pca = PCA()
        pca_features = pca.fit_transform(self.data_frame.loc[:, scaled_columns])
        component_names = [f"PC{i + 1}" for i in range(pca_features.shape[1])]
        pca_features = pd.DataFrame(pca_features, columns=component_names)
        if handel == 'merge':
            # Merge PCA features with original dataframe
            self.data_frame.reset_index(inplace=True, drop=False)
            self.data_frame = pd.concat([self.data_frame, pca_features], axis=1)
            self.data_frame.set_index(index, inplace=True)
        elif handel == 'replace':
            # Replace original features with PCA features
            self.data_frame = pd.concat([self.data_frame['class'], pca_features], axis=1)
        else:
            # Return PCA features only
            self.data_frame = pca_features

    def extend_data_by_k_means(self, features, numbers_of_cluster):
        # Apply KMeans clustering and extend dataframe with cluster information
        data = self.data_frame.copy()
        data.reset_index(inplace=True, drop=True)
        data_selected = data.loc[:, features].astype(np.float32)
        k_means_scores = list()
        for number in numbers_of_cluster:
            k_means = KMeans(number, random_state=42)
            y = k_means.fit_predict(data_selected)
            score = silhouette_score(data_selected, y)
            k_means_scores.append([k_means, score, y])
        k_means_scores = sorted(k_means_scores, key=lambda x: x[1], reverse=True)
        k_means = k_means_scores.pop()
        self.data_frame["cluster"] = k_means[2]

    def calculate_mutual_inf_class(self, target, number_of_features):
        # Calculate and return mutual information scores for features
        features = self.data_frame.copy()
        features.reset_index(inplace=True, drop=True)
        object_columns = features.select_dtypes(include='object')
        features.drop(columns=object_columns, inplace=True)
        discrete_features = features.select_dtypes(include='number')
        mi_scores = mutual_info_classif(y=target, X=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=features.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores[:number_of_features]

    def calculate_mutual_inf_regression(self, target, number_of_features):
        features = self.data_frame.copy()
        features.reset_index(inplace=True, drop=True)
        object_columns = features.select_dtypes(include='object')
        features.drop(columns=object_columns, inplace=True)
        continues_features = features.select_dtypes(include='number')
        mi_scores = mutual_info_regression(y=target, X=continues_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=features.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores[:number_of_features]


