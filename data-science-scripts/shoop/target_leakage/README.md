# Target Leakage research & experimentation
**Last updated**: 1/31/2020. Maintained by Alex Shoop (@Shoop on Slack).

## DR_dev Workflow (R&D Engineering-only)
- `python DR_dev/get_ace2.py {--all|--one}`

In order to run the scripts in `DR_dev`, you will need a `(datarobot-X.X)` developer environment.

Make sure to install the DR Python Client in the dev env if you haven't already: `pip install datarobot`

Finally, don't forget to set the correct Python path, such as: `export PYTHONPATH=/home/vagrant/workspace/DataRobot`

**NB**: The config file `drconfig.yaml` in vagrant should have the endpoint: `http://localhost/api/v2`

## DR_pyclient Workflow
- `python DR_pyclient/create_projects.py <INPUT_ARG>` (running with no INPUT_ARG will tell you what is allowed)
- `python DR_pyclient/make_dogfooding_dataset.py`
- `python DR_pyclient/create_dogfooding_project.py`
- `python DR_pyclient/delete_projects.py`
