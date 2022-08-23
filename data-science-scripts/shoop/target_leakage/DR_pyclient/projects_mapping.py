"""Project dict mapping with dataset names, target names, and target types."""

import os

USER = os.path.expanduser("~/")

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
