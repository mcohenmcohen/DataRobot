"""Create a DR project using the dogfooding dataset to predict 'is_leak'."""
import os

import datarobot as dr

# DR Client setup
USER = os.path.expanduser("~/")
client = dr.Client(config_path="{}{}".format(USER, ".config/datarobot/drconfig.yaml"))

# Dogfooding dataset filepath
dogfooding_dataset = os.path.dirname(os.path.realpath(__file__)) + "/target_leakage_dogfooding.csv"
dataset_target = "is_leak"
proj_name = "target_leakage_dogfooding"

# DR Project (Autopilot Manual)
proj = dr.Project.start(
    sourcedata="{}".format(dogfooding_dataset),
    target=dataset_target,
    project_name=proj_name,
    worker_count=4,
    autopilot_on=False,
)
