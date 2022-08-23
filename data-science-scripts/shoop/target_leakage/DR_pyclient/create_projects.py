"""Create DR projects from user input arguments. (Initial code borrowed from Jett)"""
import os
import sys
from datetime import datetime

import datarobot as dr

# DR Client setup
USER = os.path.expanduser("~/")
client = dr.Client(config_path="{}{}".format(USER, ".config/datarobot/drconfig.yaml"))

# Local path (for now)
LOCAL_FILEPATH = (
    USER + "workspace/target-leakage-research/Target_leakage_example_datasets/"
)

PROJECT_DICT = {
    "LEAKAGE_ANOMALY": {
        "name": "Anomaly-KDDTrain20.csv",
        "target": "target",
        "type": "Binary",
        "target_type": "Binary",
    },
    "LEAKAGE_DIAMOND": {
        "name": "diamonds_80.csv",
        "target": "price",
        "type": "Regression",
        "target_type": "Regression",
    },
    "LEAKAGE_LC": {
        "name": "DR_Demo_LendingClub_Guardrails.csv",
        "target": "is_bad",
        "type": "Binary",
        "target_type": "Binary",
    },
    "LEAKAGE_MOTO": {
        "name": "Motorcycle_insurance_claims_leak.csv",
        "target": "TotalClaimCost",
        "type": "Regression",
        "target_type": "Regression",
    },
    "LEAKAGE_NBA": {
        "name": "NBA_shot_logs.csv",
        "target": "SHOT_RESULT",
        "type": "Binary",
        "target_type": "Binary",
    },
    "LEAKAGE_PHONE": {
        "name": "phone_gender_narrow_leak.csv",
        "target": "gender",
        "type": "Binary",
        "target_type": "Binary",
    },
}

ARG_MAP = {
    "ALL": [
        "LEAKAGE_ANOMALY", "LEAKAGE_DIAMOND", "LEAKAGE_LC",
        "LEAKAGE_MOTO", "LEAKAGE_NBA", "LEAKAGE_PHONE",
    ],
    "anomaly": ["LEAKAGE_ANOMALY"],
    "diamond": ["LEAKAGE_DIAMOND"],
    "lc": ["LEAKAGE_LC"],
    "moto": ["LEAKAGE_MOTO"],
    "nba": ["LEAKAGE_NBA"],
    "phone": ["LEAKAGE_PHONE"],
}


def get_available_metrics(file_location=LOCAL_FILEPATH, dataset_name=None, dataset_target=None):
    """Gets the list of available metrics (ignoring Weighted and TS) for a dataset & target"""

    proj = dr.Project.create(
        sourcedata="{}{}".format(file_location, dataset_name)
    )
    available_metrics = proj.get_metrics(dataset_target)["available_metrics"]
    time_series_only = ("Theil's U", "MASE")
    experimentation_metrics = ('LogLoss', 'RMSE', 'Gini Norm')
    proj.delete()
    return [
        metric for metric in available_metrics
        if ("Weighted" not in metric and metric not in time_series_only and metric in experimentation_metrics)
    ]


def start_projects(project_list, autopilot_setting, file_location=LOCAL_FILEPATH):
    """Creates each project in the list of projects specified by user args"""

    # Must have projects
    if not project_list:
        raise ValueError(
            "Project list is empty. Please specify args:\n{}".format(list(ARG_MAP))
        )

    # Iterate through each project
    for project in project_list:
        # Save project options
        proj_options = PROJECT_DICT[project]

        # Naming
        dataset_name = proj_options["name"]

        # Target settings
        dataset_target = proj_options["target"]
        dataset_target_type = proj_options["target_type"]

        # Need to get available metrics
        print("Available project metrics:\n")
        print("--------------------------\n")
        available_metrics = get_available_metrics(file_location, dataset_name, dataset_target)
        print(available_metrics)

        # Iterate through each metric and create a DR Project for each one
        for proj_metric in available_metrics:
            # More naming
            proj_name = "({}) - {}".format(proj_metric, dataset_name)

            # Print status messages
            print("Starting New Project: {}\n".format(project))
            print("    Project Name: {}".format(proj_name))
            print("    Target: {}".format(dataset_target))
            print("    Target Type: {}".format(dataset_target_type))
            print("    Metric: {}".format(proj_metric))

            # Start a DR Project with Manual Autopilot
            new_proj = dr.Project.start(
                sourcedata="{}{}".format(file_location, dataset_name),
                target=dataset_target,
                project_name=proj_name,
                metric=proj_metric,
                worker_count=4,
                target_type=dataset_target_type,
                autopilot_on=autopilot_setting,
            )
            pid = new_proj.id
            print("    Project ID: {}\n".format(pid))

            # Project created status message
            print("Project creation completed!\n")
            print("---------------------------\n")

if __name__=="__main__":
    # Print client endpoint
    start_time = datetime.now()
    print("\n\nClient Endpoint:")
    print("----------------\n{}\n".format(client.endpoint))

    # Save user args from command line
    user_args = sys.argv[1:]

    autopilot_toggle = False
    if "--autopilot-on" in user_args:
        autopilot_toggle = True
        try:
            user_args.remove("--autopilot-on")
        except ValueError:
            pass

    # Add all projects based on the arguments mapping
    project_list = set()
    try:
        for arg in user_args:
            project_list = project_list.union(ARG_MAP[arg])
    except KeyError as e:
        string = "{}\nArgs must be from the following list:\n{}".format(e, list(ARG_MAP))
        raise ValueError(string)

    # START ALL PROJECTS
    print("\nProject List:")
    print("-------------\n{}\n".format(list(project_list)))
    print("\nProject Status Updates:")
    print("-----------------------\n")

    start_projects(project_list, autopilot_toggle)

    end_time = datetime.now()
    time_taken = str(end_time - start_time)
    print("Time taken: " + time_taken)
