#!/usr/bin/python

import os
import fnmatch
import subprocess

# https://stackoverflow.com/a/16465439
path = "/home/datarobot/workspace/DataRobot"
for dirpath, subdirs, files in os.walk(path):
    for x in files:
        if x.endswith(".py") and not x.endswith(".pyc"):
            subprocess.call(["eradicate", "--in-place", os.path.join(dirpath, x)])
