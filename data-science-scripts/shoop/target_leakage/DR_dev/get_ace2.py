# Get normalized ACE scores (or Robust ace score if FF enabled) and
# threshold-pair results for leaky datasets.

import argparse
import os
import sys
from datetime import datetime

import boto3
import datarobot as dr
import pandas as pd
import pymongo
import yaml
from bson import ObjectId


def get_experimentation_metrics(available_metrics=None):
    """
    Gets the list of experimentation metrics from available metrics.
    Ignoring Time Series metrics for now as well.
    """
    time_series_only = ('Theil\'s U', 'MASE')
    experimentation_metrics = (
        'Accuracy',
        'AUC',
        'FVE Binomial',
        'Gamma Deviance',
        'Gini',
        'Gini Norm',
        'Kolmogorov-Smirnov',
        'LogLoss',
        'Poisson Deviance',
        'MAD',
        'MAE',
        'MAPE',
        'RMSE',
        'R Squared',
        'Tweedie Deviance',
    )
    return [
        metric
        for metric in available_metrics
        if (
            'Weighted' not in metric
            and metric not in time_series_only
            and metric in experimentation_metrics
        )
    ]


def main(argv=None):
    start_time = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--all', action='store_true', help="Build DR Projects and get ACE scores for ALL datasets"
    )
    parser.add_argument(
        '--one', action='store_true', help="Build DR Projs and get ACE for LendingClub dataset"
    )
    args = parser.parse_args(argv)
    if (not args.all and not args.one) or (args.all and args.one):
        print(
            "No arguments specified or too many arguments. Use '--help' to see available arguments."
        )
        return 0

    # DR Client setup and MongoDB connection (NOTE: requires running local application)
    USER = os.path.expanduser('~/')
    client = dr.Client(config_path='{}{}'.format(USER, '.config/datarobot/drconfig.yaml'))
    mongo_conn = pymongo.MongoClient()
    eda_collection = mongo_conn['MMApp']['eda']

    # Read yaml file containing datasets with known target leakage
    datasets_yaml_file = os.path.realpath(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets_leakage.yaml')
    )
    with open(datasets_yaml_file, 'r') as f:
        datasets_dict = yaml.safe_load(f)

    # Prepare dataframe to output ACE results
    results_columns = [
        'dataset_name',
        'target_name',
        'feature_name',
        'optimization_metric',
        'ace_transformation',
        'ace_normalized_score',
        'ace_target_leakage_score',
        'moderate_threshold',
        'high_threshold',
        'leakage_type',
        'is_leak_actual',
    ]
    results_df = pd.DataFrame(columns=results_columns)
    unsupported_regression_metrics = ('Accuracy', 'AUC', 'LogLoss')
    unsupported_multiclass_metrics = (
        'Gamma Deviance',
        'Gini Norm',
        'MAE',
        'Poisson Deviance',
        'RMSE',
        'R Squared',
        'Tweedie Deviance',
    )

    if args.all:
        # Iterate through all loaded datasets
        for dataset in datasets_dict:
            print("##################################################")
            dataset_filename = dataset['dataset_name']
            dataset_name = dataset_filename.split('/')[-1]
            target = dataset['target']
            target_type = dataset.get('target_type')
            leaky_features = dataset.get('leaky_features')
            print("Dataset: {}".format(dataset_name))
            print("Target: {}".format(target))
            print("Potential leaks: {}".format(leaky_features))

            if 'http' not in dataset_filename:  # local data file and not S3
                dataset_filename = os.path.realpath(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), dataset_filename)
                )
            if 'private' in dataset_filename:  # Requires an AWS access key and secret access key
                s3_file = dataset_name
                s3 = boto3.client('s3')
                with open(s3_file, 'wb') as data:
                    s3.download_fileobj(
                        'datarobot-private-datasets-nonredistributable', dataset_name, data
                    )
                dataset_filename = s3_file

            # Need to get available metrics and then get experimentation metrics
            temp_proj = dr.Project.create(sourcedata=dataset_filename)
            temp_eda_dict = eda_collection.find_one({'pid': ObjectId(temp_proj.id)})['eda']
            available_metrics = [
                metric['short_name'] for metric in temp_eda_dict[target]['metric_options']['all']
            ]
            experimentation_metrics = get_experimentation_metrics(available_metrics)
            print("Available experimentation metrics: {}".format(experimentation_metrics))
            temp_proj.delete()

            for metric in experimentation_metrics:
                if (target_type == 'Multiclass' and metric in unsupported_multiclass_metrics) or (
                    target_type == 'Regression' and metric in unsupported_regression_metrics
                ):
                    out = "This is a {proj_type} project. {metric} is not supported.".format(
                        proj_type=target_type, metric=metric,
                    )
                    print(out)
                    print("Continuing to next experimentation metric...")
                    continue
                print("##################################################")
                project_name = '({}) {}'.format(metric, dataset_name)
                print("Project name: {}".format(project_name))
                proj = dr.Project.start(
                    sourcedata=dataset_filename,
                    target=target,
                    project_name=project_name,
                    metric=metric,
                    worker_count=4,
                    target_type=target_type,
                    autopilot_on=False,
                )
                print("Project ID: {}".format(proj.id))
                eda_dict = eda_collection.find_one({'pid': ObjectId(proj.id)})['eda']

                features = proj.get_features()
                for feature in features:
                    feature_name = feature.name
                    is_leak_actual = False
                    if leaky_features is not None:
                        if feature_name in leaky_features:
                            is_leak_actual = True

                    feature_ace_results = []
                    try:
                        feature_raw_ace = eda_dict[feature_name]['profile']['raw_info']
                        feature_normalized_ace = eda_dict[feature_name]['profile']['info']
                        feature_target_leakage_ace_score = eda_dict[feature_name][
                            'target_leakage_metadata'
                        ]['importance']['value']
                        feature_leakage_type = eda_dict[feature_name]['target_leakage']

                        for pairs_and_leaks in eda_dict[feature_name][
                            'target_leakage_metadata_thresholds'
                        ]['ace_leakage_thresholds']:
                            moderate_threshold, high_threshold = pairs_and_leaks['threshold_pair']
                            detected_leakage_type = pairs_and_leaks['leakage_type']

                            ace_detected_threshold_result = [
                                dataset_name,
                                target,
                                feature_name,
                                metric,
                                'normalized_wrt_target',
                                feature_normalized_ace,
                                feature_target_leakage_ace_score,
                                moderate_threshold,
                                high_threshold,
                                detected_leakage_type,
                                is_leak_actual,
                            ]
                            print(ace_detected_threshold_result)
                            feature_ace_results.append(ace_detected_threshold_result)
                    except KeyError:  # use arbitrary values
                        feature_raw_ace = 0.0
                        feature_normalized_ace = 1.0
                        feature_target_leakage_ace_score = 1.0
                        feature_leakage_type = False

                        ace_detected_threshold_result = [
                            dataset_name,
                            target,
                            feature_name,
                            metric,
                            'normalized_wrt_target',
                            feature_normalized_ace,
                            feature_target_leakage_ace_score,
                            0.80,
                            0.975,
                            feature_leakage_type,
                            is_leak_actual,
                        ]
                        print(ace_detected_threshold_result)
                        feature_ace_results.append(ace_detected_threshold_result)

                    results_rows = pd.DataFrame(feature_ace_results, columns=results_columns)
                    results_df = results_df.append(results_rows, ignore_index=True)
    if args.one:
        # Do the above, but for only LendingClub dataset
        print("##################################################")
        lc_dataset = datasets_dict[0]
        dataset_filename = lc_dataset['dataset_name']
        dataset_name = dataset_filename.split('/')[-1]
        target = lc_dataset['target']
        target_type = lc_dataset.get('target_type')
        leaky_features = lc_dataset.get('leaky_features')
        print("Dataset: {}".format(dataset_name))
        print("Target: {}".format(target))
        print("Potential leaks: {}".format(leaky_features))

        if 'http' not in dataset_filename:  # local data file and not S3
            dataset_filename = os.path.realpath(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), dataset_filename)
            )
        if 'private' in dataset_filename:  # Requires an AWS access key and secret access key
            s3_file = dataset_name
            s3 = boto3.client('s3')
            with open(s3_file, 'wb') as data:
                s3.download_fileobj(
                    'datarobot-private-datasets-nonredistributable', dataset_name, data
                )
            dataset_filename = s3_file

        # Need to get available metrics and then get experimentation metrics
        temp_proj = dr.Project.create(sourcedata=dataset_filename)
        temp_eda_dict = eda_collection.find_one({'pid': ObjectId(temp_proj.id)})['eda']
        available_metrics = [
            metric['short_name'] for metric in temp_eda_dict[target]['metric_options']['all']
        ]
        experimentation_metrics = get_experimentation_metrics(available_metrics)
        print("Available experimentation metrics: {}".format(experimentation_metrics))
        temp_proj.delete()

        for metric in experimentation_metrics:
            if (target_type == 'Multiclass' and metric in unsupported_multiclass_metrics) or (
                target_type == 'Regression' and metric in unsupported_regression_metrics
            ):
                out = "This is a {proj_type} project. {metric} is not supported.".format(
                    proj_type=target_type, metric=metric,
                )
                print(out)
                print("Continuing to next experimentation metric...")
                continue
            print("##################################################")
            project_name = '({}) {}'.format(metric, dataset_name)
            print("Project name: {}".format(project_name))
            proj = dr.Project.start(
                sourcedata=dataset_filename,
                target=target,
                project_name=project_name,
                metric=metric,
                worker_count=4,
                target_type=target_type,
                autopilot_on=False,
            )
            print("Project ID: {}".format(proj.id))
            eda_dict = eda_collection.find_one({'pid': ObjectId(proj.id)})['eda']
            features = proj.get_features()
            for feature in features:
                feature_name = feature.name
                is_leak_actual = False
                if leaky_features is not None:
                    if feature_name in leaky_features:
                        is_leak_actual = True

                feature_ace_results = []
                try:
                    feature_raw_ace = eda_dict[feature_name]['profile']['raw_info']
                    feature_normalized_ace = eda_dict[feature_name]['profile']['info']
                    feature_target_leakage_ace_score = eda_dict[feature_name][
                        'target_leakage_metadata'
                    ]['importance']['value']
                    feature_leakage_type = eda_dict[feature_name]['target_leakage']

                    for pairs_and_leaks in eda_dict[feature_name][
                        'target_leakage_metadata_thresholds'
                    ]['ace_leakage_thresholds']:
                        moderate_threshold, high_threshold = pairs_and_leaks['threshold_pair']
                        detected_leakage_type = pairs_and_leaks['leakage_type']

                        ace_detected_threshold_result = [
                            dataset_name,
                            target,
                            feature_name,
                            metric,
                            'normalized_wrt_target',
                            feature_normalized_ace,
                            feature_target_leakage_ace_score,
                            moderate_threshold,
                            high_threshold,
                            detected_leakage_type,
                            is_leak_actual,
                        ]
                        print(ace_detected_threshold_result)
                        feature_ace_results.append(ace_detected_threshold_result)
                except KeyError:  # use arbitrary values
                    feature_raw_ace = 0.0
                    feature_normalized_ace = 1.0
                    feature_target_leakage_ace_score = 1.0
                    feature_leakage_type = False

                    ace_detected_threshold_result = [
                        dataset_name,
                        target,
                        feature_name,
                        metric,
                        'normalized_wrt_target',
                        feature_normalized_ace,
                        feature_target_leakage_ace_score,
                        0.80,
                        0.975,
                        feature_leakage_type,
                        is_leak_actual,
                    ]
                    print(ace_detected_threshold_result)
                    feature_ace_results.append(ace_detected_threshold_result)

                results_rows = pd.DataFrame(feature_ace_results, columns=results_columns)
                results_df = results_df.append(results_rows, ignore_index=True)

    # Saving results
    results_filepath = os.path.realpath(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ace_results.csv')
    )
    results_df.to_csv(results_filepath, index=None, header=True)
    time_taken = str(datetime.now() - start_time)
    print("Time taken: {}".format(time_taken))
    return 0


if __name__ == '__main__':
    sys.exit(main())
