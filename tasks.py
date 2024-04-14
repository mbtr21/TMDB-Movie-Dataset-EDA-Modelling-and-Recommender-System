from celery import shared_task
from config import celery_init_app
from flask import Flask
from models import MachineLearningClassifier
from pre_processor import FeatureSelection
import pandas as pd
import multiprocessing

server = Flask(__name__)
server.config.from_mapping(
    CELERY=dict(
        broker_url='amqp://guest:guest@localhost//',
        result_backend='rpc://',
        task_ignore_result=True,
    ),
)

celery_app = celery_init_app(server)


@shared_task(ignore_result=False)
def generate_bar_chart(data_frame, radio_item, movie_item):
    df = data_frame[data_frame[f'{radio_item}'].isin(movie_item)]
    df = df.groupby(f'{radio_item}').agg({'budget': 'mean', 'popularity': 'mean', 'vote_average': 'mean',
                                          'runtime': 'mean', 'revenue': 'mean', 'vote_count': 'mean'})
    df.reset_index(inplace=True)
    return df


@shared_task(ignore_result=False)
def machine_learning(data_frame, log_dropdown, target, classifier_items):
    feature_selector = FeatureSelection(data_frame)
    features = feature_selector.data_frame.select_dtypes(include=['number']).columns
    if classifier_items == 'k_means':
        feature_selector.extend_data_by_k_means(features=features, numbers_of_cluster=[5])
    # p3.start()
    # p3.join()
    # if classifier_items == 'dbscan':
    # p4 = multiprocessing.Process(feature_selector.extend_data_by_hdbscan(features=features))
    # p4.start()
    # p4.join()

    feature_selector.add_log_transform(columns=log_dropdown)
    feature_selector.reduction_dimension_by_pca(scaled_columns=features)
    # feature_selector.calculate_mutual_inf_class(target=target, number_of_features=12)

    data = list()
    model = MachineLearningClassifier(data_frame=feature_selector.data_frame, target=target)
    model.train_test_split()
    model.fit_xgboost_classifier()
    data.append(model.metrics())
    model.fit_logistic_regression_classifier()
    data.append(model.metrics())
    model.fit_decision_tree_classifier()
    data.append(model.metrics())
    model.fit_random_forest_classifier()
    data.append(model.metrics())

    df = pd.DataFrame(data=data)
    return df
