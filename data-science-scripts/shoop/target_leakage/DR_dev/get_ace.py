"""Get ACE scores for leaky datasets."""
import argparse
import os
import sys
from datetime import datetime

from bson import ObjectId
import yaml

import datarobot as dr
import pandas as pd
import pymongo


def get_experimentation_metrics(available_metrics=None):
    """
    Gets the list of experimentation metrics from available metrics.
    Ignoring Time Series metrics for now as well.
    """
    time_series_only = ('Theil\'s U', 'MASE')
    experimentation_metrics = (
        'AUC',
        'FVE Binomial',
        'Gamma Deviance',
        'Gini Norm',
        'LogLoss',
        'RMSE',
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
        print("No arguments specified or too many arguments. Use '--help' to see available arguments.")
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
        'ace_score',
        'leakage_type',
        'is_leak_actual',
    ]
    results_df = pd.DataFrame(columns=results_columns)

    if args.all:
        # Iterate through all loaded datasets
        for dataset in datasets_dict:
            print("##################################################")
            dataset_filename = dataset['dataset_name']
            dataset_name = dataset_filename.split('/')[-1]
            target = dataset['target']
            target_type = dataset['target_type']
            leaky_features = dataset['leaky_features']
            print("Dataset: {}".format(dataset_name))
            print("Target: {}".format(target))
            print("Potential leaks: {}".format(leaky_features))

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

                print("##################################################")
                features = proj.get_features()
                for feature in features:
                    feature_name = feature.name
                    is_leak_actual = False
                    if feature_name in leaky_features:
                        is_leak_actual = True

                    try:
                        feature_raw_info = eda_dict[feature_name]['profile']['raw_info']
                        feature_info = eda_dict[feature_name]['profile']['info']
                        leakage_type = eda_dict[feature_name]['target_leakage']
                    except KeyError:  # use arbitrary values
                        feature_raw_info = 0.0
                        feature_info = 1.0

                    results_list_raw = [
                        dataset_name,
                        target,
                        feature_name,
                        metric,
                        'raw',
                        feature_raw_info,
                        leakage_type,
                        is_leak_actual,
                    ]
                    results_list_normalized = [
                        dataset_name,
                        target,
                        feature_name,
                        metric,
                        'normalized_wrt_target',
                        feature_info,
                        leakage_type,
                        is_leak_actual,
                    ]
                    print(results_list_raw)
                    print(results_list_normalized)
                    results_df_rows = pd.DataFrame([results_list_raw, results_list_normalized], columns=results_columns)
                    results_df = results_df.append(results_df_rows, ignore_index=True)

    if args.one:
        # Do the above, but for only LendingClub dataset
        print("##################################################")
        lc_dataset = datasets_dict[0]
        dataset_filename = lc_dataset['dataset_name']
        dataset_name = dataset_filename.split('/')[-1]
        target = lc_dataset['target']
        target_type = lc_dataset['target_type']
        leaky_features = lc_dataset['leaky_features']
        print("Dataset: {}".format(dataset_name))
        print("Target: {}".format(target))
        print("Potential leaks: {}".format(leaky_features))

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

            print("##################################################")
            features = proj.get_features()
            for feature in features:
                feature_name = feature.name
                is_leak_actual = False
                if feature_name in leaky_features:
                    is_leak_actual = True

                try:
                    feature_raw_info = eda_dict[feature_name]['profile']['raw_info']
                    feature_info = eda_dict[feature_name]['profile']['info']
                    leakage_type = eda_dict[feature_name]['target_leakage']
                except KeyError:  # use arbitrary values
                    feature_raw_info = 0.0
                    feature_info = 1.0

                results_list_raw = [
                    dataset_name,
                    target,
                    feature_name,
                    metric,
                    'raw',
                    feature_raw_info,
                    leakage_type,
                    is_leak_actual,
                ]
                results_list_normalized = [
                    dataset_name,
                    target,
                    feature_name,
                    metric,
                    'normalized_wrt_target',
                    feature_info,
                    leakage_type,
                    is_leak_actual,
                ]
                print(results_list_raw)
                print(results_list_normalized)
                results_df_rows = pd.DataFrame([results_list_raw, results_list_normalized], columns=results_columns)
                results_df = results_df.append(results_df_rows, ignore_index=True)

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
