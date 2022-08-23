# Getting presentation data for DS dev sync

import os
from pathlib import Path
from pprint import pprint
from datetime import datetime

import boto3
import datarobot as dr
import pandas as pd
import yaml


def main():
    cwd = Path.cwd()
    leaky_datasets_file = f'{cwd.parent.parent}/target_leakage/datasets_leakage.yaml'

    # DR Client setup
    USER = os.path.expanduser('~/')
    client = dr.Client(config_path=f'{USER}.config/datarobot/drconfig.yaml')

    # Read datasets yaml to make datasets dict
    with open(leaky_datasets_file, 'r') as f:
        list_of_datasets = yaml.safe_load(f)

    # Setting up results
    results_columns = (
        'dataset_name',
        'number_of_features',
        'leaky_features_actual',
        'new_tld_high_risk_leaks',
        'new_tld_moderate_risk_leaks',
        'old_tld_high_risk_leaks',
        'old_tld_moderate_risk_leaks',
    )
    list_of_results = []

    # DATASETS LOOP
    for dataset in list_of_datasets:
        results_dict = {}

        # dataset yaml values
        # SAVE: dataset_name
        dataset_source = dataset['dataset_name']
        dataset_name = dataset_source.split('/')[-1]
        dataset_target = dataset['target']
        target_type = dataset.get('target_type')
        leaky_features = dataset.get('leaky_features')
        results_dict['dataset_name'] = dataset_name

        # AWS private dataset access
        if 'private' in dataset_source:  # Requires an AWS access key and secret access key
            s3_file = dataset_name
            s3 = boto3.client('s3')
            with open(s3_file, 'wb') as data:
                s3.download_fileobj(
                    'datarobot-private-datasets-nonredistributable', dataset_name, data
                )
            dataset_source = s3_file

        # DR Project start
        print("##################################################")
        print(f"Project name: {dataset_name}")
        proj = dr.Project.start(
            sourcedata=dataset_source,
            target=dataset_target,
            project_name=dataset_name,
            worker_count=4,
            target_type=target_type,
            autopilot_on=False,  # AUTOPILOT MANUAL
        )
        print(f"Project ID: {proj.id}")

        # Project features
        features = proj.get_features()
        features_count = len(features)
        print(f"Number of features: {features_count}")
        results_dict['number_of_features'] = features_count

        # Leakage information
        leakage_dict = {
            'new_tld_high_risks': 0,
            'new_tld_mod_risks': 0,
            'old_tld_high_risks': 0,
            'old_tld_mod_risks': 0,
        }
        for feature in features:
            # skip target
            if feature.name == dataset_target:
                continue
            # New TLD
            if feature.target_leakage == 'HIGH_RISK':
                leakage_dict['new_tld_high_risks'] += 1
            if feature.target_leakage == 'MODERATE_RISK':
                leakage_dict['new_tld_mod_risks'] += 1
            # Old TLD
            if feature.importance is not None:
                if feature.importance >= 0.975:
                    leakage_dict['old_tld_high_risks'] += 1
                    continue
                if feature.importance >= 0.85:
                    leakage_dict['old_tld_mod_risks'] += 1

        print(f"Detected leaky features, New TLD (High Risk): {leakage_dict['new_tld_high_risks']}")
        print(
            f"Detected leaky features, New TLD (Moderate Risk): {leakage_dict['new_tld_mod_risks']}"
        )
        print(f"Detected leaky features, Old TLD (High Risk): {leakage_dict['old_tld_high_risks']}")
        print(
            f"Detected leaky features, Old TLD (Moderate Risk): {leakage_dict['old_tld_mod_risks']}"
        )
        results_dict['leaky_features_actual'] = len(leaky_features)
        results_dict['new_tld_high_risk_leaks'] = leakage_dict['new_tld_high_risks']
        results_dict['new_tld_moderate_risk_leaks'] = leakage_dict['new_tld_mod_risks']
        results_dict['old_tld_high_risk_leaks'] = leakage_dict['old_tld_high_risks']
        results_dict['old_tld_moderate_risk_leaks'] = leakage_dict['old_tld_mod_risks']

        # add results to final output list
        list_of_results.append(results_dict)

    # Save final output df
    results_df = pd.DataFrame(list_of_results, columns=results_columns)
    results_df.to_csv('ds_sync_results.csv', header=True, index=False)


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    time_taken = datetime.now() - start_time
    print(f"Time taken: {time_taken.seconds // 60} mins")
