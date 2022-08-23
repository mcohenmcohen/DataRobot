import argparse
import json
import requests
import time


def main(api_key,  eu_endpoint=False, project_id=None):
    # Credentials (PROD)
    API_KEY = api_key
    headers = {"Authorization": f"Bearer {API_KEY}", "content-type": "application/json"}
    if eu_endpoint:
        api_endpoint = "https://app.eu.datarobot.com/api/v2"
    else:
        api_endpoint = "https://app.datarobot.com/api/v2"

    selected_project = None
    supported_target_types = {"Binary", "Regression", "Multiclass"}
    projects_endpoint = f"{api_endpoint}/projects/"

    if project_id:
        print(f"Making sure {project_id} is a valid project...")
        time.sleep(1)
        project_endpoint = f"{projects_endpoint}/{project_id}/"
        project_response = requests.get(url=project_endpoint, headers=headers)
        selected_project = project_response.json()
        assert project_response.status_code == 200, selected_project

        assert selected_project.get("autopilotMode") is not None
        assert not selected_project.get("unsupervisedMode")
        assert selected_project.get("partition", {}).get("useTimeSeries") is None
        assert selected_project.get("targetType") in supported_target_types
        print(f"Done getting project!")
        print("-------------------------------------------------------")

    else:
        print("No Project ID set, getting recent projects...")
        time.sleep(1)

        projects_response = requests.get(url=projects_endpoint, headers=headers)
        all_projects = projects_response.json()
        assert projects_response.status_code == 200, all_projects

        print("Done getting Recent Projects!")
        print("-------------------------------------------------------")

        # Filter to only supported projects
        print("Filtering to only supported projects...")
        time.sleep(2)
        supported_projects = []

        for project in all_projects:
            # Skip if project has not started Autopilot
            # NOTE: when value is 0, that means non-Manual Autopilot
            if project.get("autopilotMode") is None:
                continue
            # Skip unsupervised projects
            if project.get("unsupervisedMode"):
                continue
            # Skip TS projects
            if project.get("partition", {}).get("useTimeSeries"):
                continue
            # Only support for Binary Classification, Regression, and Multiclass
            if project.get("targetType") in supported_target_types:
                supported_projects.append(project)
                continue

        print("Done with filtering!")
        print("-------------------------------------------------------")
        print("These are the 5 most recent projects that are supported:")
        for project in supported_projects[:5]:
            print(project["projectName"])

    project_id = selected_project["id"]
    project_name = selected_project["projectName"]
    print(f"Using project: {project_name}")
    time.sleep(2)

    # Get dataset_id (i.e. featurelist ID) for most recent project
    featurelists_endpoint = f"{projects_endpoint}/{project_id}/featurelists/"
    featurelists_response = requests.get(url=featurelists_endpoint, headers=headers)
    all_featurelists = featurelists_response.json()
    assert featurelists_response.status_code == 200, all_featurelists

    # Use dataset_id with most trained models
    sorted_featurelists = sorted(all_featurelists, key=lambda fl: fl["numModels"], reverse=True)
    dataset_id = sorted_featurelists[0]["id"]
    featurelist_name = sorted_featurelists[0]["name"]
    print(f"Using feature list with most trained models: {featurelist_name}")
    time.sleep(2)

    print("-------------------------------------------------------")

    # Run MetaLearning Blueprint Search flow
    print("Kicking off MetaLearning Blueprint Search...")
    time.sleep(2)
    request_payload = {"datasetId": dataset_id}
    bp_search_endpoint = f"{projects_endpoint}/{project_id}/searchedBlueprints/"
    bp_search_response = requests.post(
        url=bp_search_endpoint, data=json.dumps(request_payload), headers=headers
    )
    assert bp_search_response.status_code == 202, bp_search_response.json()
    base_url = api_endpoint.split("/api")[0]
    print("Successfully kicked off MetaLearning Blueprint Search!")
    print("New blueprints will be added to Repository and automatically trained to Leaderboard. Project link below:")
    print(f"{base_url}/projects/{project_id}/models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MetaLearning Blueprint Search on most recent supported project")
    parser.add_argument("--api_key", help="API authorization key", required=True)
    parser.add_argument("--eu_endpoint", help="[OPTIONAL] Use EU Prod API endpoint (True/False)", default=False)
    parser.add_argument(
        "--project_id",
        help="[OPTIONAL] A specific project ID to run blueprint search on. "
        "If not specified, will use most recent supported project.",
    )
    args = parser.parse_args()
    main(api_key=args.api_key, eu_endpoint=args.eu_endpoint, project_id=args.project_id)
