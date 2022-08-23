"""Experimentation script.

Python3 script to get values of multiclass_ace() function from the local application.

Usage::

    python target_leakage_multiclass_research.py

"""

import os
import copy
from pathlib import Path
from pprint import pprint
from datetime import datetime

import boto3
import datarobot as dr
import pandas as pd
import pymongo
import yaml
from bson import ObjectId


def main():
    cwd = Path.cwd()
    leaky_multiclass_datasets = f'{cwd}/datasets_multiclass_leakage.yaml'

    # DR Client setup
    USER = os.path.expanduser('~/')
    client = dr.Client(config_path=f'{USER}.config/datarobot/drconfig.yaml')
    # local application mongo connection
    mongo_conn = pymongo.MongoClient()
    eda_collection = mongo_conn['MMApp']['eda']

    # Read datasets yaml to make datasets dict
    with open(leaky_multiclass_datasets, 'r') as f:
        list_of_datasets = yaml.safe_load(f)

    # Setting up results
    results_columns = (
        'dataset_name',
        'target_name',
        'proj_metric',
        'target_class_count',
        'feature_name',
        'target_leakage',
        'proj_metric_importance',
        'target_class_label',
        'multiclass_raw_ace_vals',
        'multiclass_normlized_ace_vals',
        'multiclass_gininorm_ace_vals',
        'multiclass_acc_ace_vals',
    )
    list_of_results = []

    # metrics for experimentation
    available_metrics = ('Accuracy', 'AUC', 'Balanced Accuracy', 'LogLoss')

    # DATASETS LOOP
    for dataset in list_of_datasets:
        # dataset yaml values
        dataset_source = dataset['dataset_name']
        dataset_name = dataset_source.split('/')[-1]
        dataset_target = dataset['target']
        target_class_count = dataset['target_class_count']
        leaky_features = dataset.get('leaky_features')

        # AWS private dataset access
        if 'private' in dataset_source:  # Requires an AWS access key and secret access key
            s3_key = dataset_source.split('.com/')[-1]
            s3_file = dataset_name.split('/')[-1]
            s3 = boto3.client('s3')
            with open(s3_file, 'wb') as data:
                s3.download_fileobj('datarobot-private-datasets-nonredistributable', s3_key, data)
            dataset_source = s3_file
            dataset_name = dataset_source

        # METRICS LOOP
        for metric in available_metrics:
            # DR Project start
            proj_name = f'{dataset_name} - {metric}'
            print("##################################################")
            print(f"Project name: {proj_name}")
            proj = dr.Project.start(
                sourcedata=dataset_source,
                target=dataset_target,
                project_name=proj_name,
                metric=metric,
                worker_count=4,
                autopilot_on=False,  # AUTOPILOT MANUAL
            )
            print(f"Project ID: {proj.id}")
            print(f"Leaky features actual: {leaky_features}")

            # access local app mongo eda collection
            eda_dict = eda_collection.find_one({'pid': ObjectId(proj.id)})['eda']

            # FEATURES LOOP
            for feature in proj.get_features():
                # skip target
                if feature.name == dataset_target:
                    continue
                # skip low_info feature, keeping feature which gets flagged as "high-risk" leak
                if True in eda_dict[feature.name]['low_info'].values():
                    if 'leakage' in eda_dict[feature.name]['low_info']:
                        if eda_dict[feature.name]['low_info']['leakage'] == False:
                            continue
                    else:
                        continue
                # skip text feature (because we don't calculate Importance in EDA2 yet)
                if eda_dict[feature.name]['types']['text'] == True:
                    continue

                # savings results of each feature
                results_dict = {}
                results_dict['dataset_name'] = dataset_name
                results_dict['target_name'] = dataset_target
                results_dict['proj_metric'] = metric
                results_dict['target_class_count'] = target_class_count
                results_dict['feature_name'] = feature.name
                results_dict['target_leakage'] = feature.target_leakage
                results_dict['proj_metric_importance'] = feature.importance

                # loop over each target-class-label
                for target_class in eda_dict[feature.name]['multiclass_raw_ace_vals']:
                    results_dict_cp = copy.deepcopy(results_dict)
                    results_dict_cp['target_class_label'] = target_class
                    results_dict_cp['multiclass_raw_ace_vals'] = eda_dict[feature.name][
                        'multiclass_raw_ace_vals'
                    ][target_class]
                    results_dict_cp['multiclass_normlized_ace_vals'] = eda_dict[feature.name][
                        'multiclass_normlized_ace_vals'
                    ][target_class]
                    results_dict_cp['multiclass_gininorm_ace_vals'] = eda_dict[feature.name][
                        'multiclass_gininorm_ace_vals'
                    ][target_class]
                    results_dict_cp['multiclass_acc_ace_vals'] = eda_dict[feature.name][
                        'multiclass_acc_ace_vals'
                    ][target_class]

                    # add results to results_list for final output df
                    list_of_results.append(results_dict_cp)

    # Save final output df
    results_df = pd.DataFrame(list_of_results, columns=results_columns)
    results_df.to_csv('tld_multiclass_experimentation.csv', header=True, index=False)


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    time_taken = datetime.now() - start_time
    print(f"Time taken: {time_taken.seconds // 60} mins")
