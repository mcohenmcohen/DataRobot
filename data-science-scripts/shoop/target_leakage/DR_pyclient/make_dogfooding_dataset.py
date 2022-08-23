"""Create the dogfooding dataset for use in a DR Project to predict 'is_leak'."""
import os

import datarobot as dr
import pandas as pd

# DR Client setup
USER = os.path.expanduser("~/")
client = dr.Client(config_path="{}{}".format(USER, ".config/datarobot/drconfig.yaml"))

# Set up the df that will have all dogfooding dataset information
column_names = [
    "dataset_name", "target", "target_type", "project_metric", "feature_name", "feature_type",
    "feature_low_info", "ace_score_normal", "leakage_risk", "is_leak",
]
dogfooding_df = pd.DataFrame(columns=column_names)

# Get full list of DR projects
projects = dr.Project.list()
for project in projects:
    # Getting feature-by-feature information
    for feature in project.get_features():
        data_dict = {}
        data_dict["dataset_name"] = project.project_name
        data_dict["target"] = project.target
        data_dict["target_type"] = project.target_type
        data_dict["project_metric"] = project.metric
        data_dict["feature_name"] = feature.name
        data_dict["feature_type"] = feature.feature_type
        data_dict["feature_low_info"] = feature.low_information
        data_dict["ace_score_normal"] = feature.importance
        data_dict["leakage_risk"] = feature.target_leakage

        # IS_LEAK
        if data_dict["leakage_risk"] in ("MODERATE_RISK", "HIGH_RISK"):
            data_dict["is_leak"] = "yes"
            # low_information is set to False for features with target leakage
            # because we want to understand the other information for the leaky feature
            data_dict["feature_low_info"] = False
        else:
            data_dict["is_leak"] = "no"

        # Add row to output dataframe
        dogfooding_df = dogfooding_df.append(data_dict, ignore_index=True)

# Put "None" for empty spaces
dogfooding_df.fillna("None", inplace=True)

# Save dogfooding dataset to CSV file
csv_filepath = os.path.dirname(os.path.realpath(__file__)) + "/target_leakage_dogfooding.csv"
dogfooding_df.to_csv(csv_filepath, index=False)
