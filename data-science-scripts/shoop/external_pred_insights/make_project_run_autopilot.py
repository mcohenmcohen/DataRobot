import requests
import json
import time

# Credentials (LOCAL)
API_KEY = ""
api_endpoint = "http://localhost/api/v2"
headers = {"Authorization": f"Bearer {API_KEY}", "content-type": "application/json"}

# Specific "10k_diabetes_train80_4cols_externalpreds.csv" dataset in AI Catalog
dataset_id = "60c3567d686520257c43ee22"

project_endpoint = f"{api_endpoint}/projects/"
project_payload = {"datasetId": dataset_id}

project = requests.post(
    url=project_endpoint, data=json.dumps(project_payload), headers=headers
)
project_id = project.json().get("pid")

# Wait for EDA1 to complete (so that we have target name for Autopilot)
for _ in range(3):
    time.sleep(1)
    project_status = requests.get(url=project.headers.get("Location"), headers=headers)
    if "projectName" in project_status.json():
        print("EDA1 is done.")
        break

# AUTOPILOT START (QUICK)
autopilot_endpoint = f"{api_endpoint}/projects/{project_id}/aim/"
autopilot_payload = {
    "target": "readmitted",
    "mode": "quick",
    "cvMethod": "user",
    "validationType": "TVH",
    "userPartitionCol": "partition_column",
    "trainingLevel": "T",
    "validationLevel": "H",
    "externalPredictions": ["Model1_Output"],
}
print(json.dumps(autopilot_payload))
start_autopilot = requests.patch(
    url=autopilot_endpoint, data=json.dumps(autopilot_payload), headers=headers
)
print(start_autopilot)
