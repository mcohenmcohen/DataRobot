"""Delete DR projects (typically used right after running create_projects.py)"""
import os

import datarobot as dr

# DR Client setup
USER = os.path.expanduser("~/")
client = dr.Client(config_path="{}{}".format(USER, ".config/datarobot/drconfig.yaml"))

# DR Projects mass deletion
projects = dr.Project.list()
for project in projects:
    print("Deleting project: " + project.project_name)
    project.delete()
