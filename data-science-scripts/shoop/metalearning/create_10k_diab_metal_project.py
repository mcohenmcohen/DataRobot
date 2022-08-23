# insert comments here

import datarobot as dr
import pandas as pd

# Credentials (LOCAL)
API_KEY = ""
api_endpoint = "http://localhost/api/v2"
headers = {"Authorization": f"Bearer {API_KEY}", "content-type": "application/json"}

# 10k_diabetes.csv dataset from AI Catalog
dataset_id = "6154d1d028ff1c3d4df8b541"

project_endpoint = f"{api_endpoint}/projects/"
project_payload = {"datasetId": dataset_id}

project = requests.post(
    url=project_endpoint, data=json.dumps(project_payload), headers=headers
)
project_id = project.json().get("pid")

# Wait for EDA1 to complete (so that we have target name for Autopilot)
for _ in range(3):
    time.sleep(2)
    project_status = requests.get(url=project.headers.get("Location"), headers=headers)
    if "projectName" in project_status.json():
        print("EDA1 is done.")
        break

# AUTOPILOT START (QUICK)
autopilot_endpoint = f"{api_endpoint}/projects/{project_id}/aim/"
autopilot_payload = {
    "target": "readmitted",
    "mode": "manual",
}
print(json.dumps(autopilot_payload))
start_autopilot = requests.patch(
    url=autopilot_endpoint, data=json.dumps(autopilot_payload), headers=headers
)
print(start_autopilot)



# 4) Then check Autopilot status if it's complete or not

# 5) Then do UI part, or try out blueprint search public API routes, like /searchedBlueprints/
# request.post("/searchedBlueprints/", json=payload)
# see what response is, and experiment from there
