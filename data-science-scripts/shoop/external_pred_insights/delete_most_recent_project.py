import datarobot as dr

# Credentials (LOCAL)
API_KEY = ""
API_ENDPOINT = "http://localhost/api/v2"

dr.Client(token=API_KEY, endpoint=API_ENDPOINT)

projects = dr.Project.list()
latest_project = projects[0]
print(f"Most recent project is {latest_project.project_name}")
print("Deleting...")
latest_project.delete()
print("Done!")
