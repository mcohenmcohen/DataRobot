import datarobot as dr
import requests
from datarobot.enums import MODEL_REPLACEMENT_REASON

# Use drcongfig yaml or hard code file:///Users/DataRobot/Downloads/html/setup/configuration.html

prediction_endpoint = 'https://datarobot-predictions.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/predictions'

USERNAME = ''
API_TOKEN = ''

# Deployment predictions
class DataRobotPredictionError(Exception):
    pass


def make_datarobot_deployment_predictions(data, deployment_id):
    """
    Make predictions on data provided using DataRobot deployment_id provided.
    See docs for details:
         https://staging.datarobot.com/docs/users-guide/deploy/api/new-prediction-api.html

    Parameters
    ----------
    data : str
        Feature1,Feature2
        numeric_value,string
    deployment_id : str
        The ID of the deployment to make predictions with.

    Returns
    -------
    Response schema: https://staging.datarobot.com/docs/users-guide/deploy/api/new-prediction-api.html#response-schema

    Raises
    ------
    DataRobotPredictionError if there are issues getting predictions from DataRobot
    """
    # Set HTTP headers. The charset should match the contents of the file.
    headers = {'Content-Type': 'text/plain; charset=UTF-8', 'datarobot-key': '27782'}
    url = 'https://datarobot-hackathon.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/predictions'.format(
        deployment_id=deployment_id
    )
    # Make API request for predictions
    predictions_response = requests.post(url, auth=(USERNAME, API_TOKEN), data=data,
                                         headers=headers)
    _raise_dataroboterror_for_status(predictions_response)
    # Return a Python dict following the schema in the documentation
    return predictions_response.json()


def _raise_dataroboterror_for_status(response):
    """Raise DataRobotPredictionError if the request fails along with the response returned"""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        errMsg = '{code} Error: {msg}'.format(code=response.status_code, msg=response.text)
        raise DataRobotPredictionError(errMsg)


# Dataset
training_url = "https://s3.amazonaws.com/datarobot_public_datasets/boston_housing_80.csv"
scoring_url=  "https://s3.amazonaws.com/datarobot_public_datasets/boston_housing_20.csv"
target = "MEDV"

# Create Project and run Autopilot
project = dr.Project.create(project_name=target, sourcedata=training_url)
project.unlock_holdout()
project.set_target(target=target, worker_count=30)
project.wait_for_autopilot(verbosity=dr.VERBOSITY_LEVEL.SILENT)

# Get model recommended for deployment
recommendation = dr.ModelRecommendation.get(project.id)
model =recommendation.get_model()

# Find Default Prediction Server
pred_server = dr.PredictionServer.list()[0].id

# Create deployment
deployment = dr.Deployment.create_from_learning_model(model.id, 'Boston Housing API Starter', default_prediction_server_id=pred_server)

# Check model health. Should currently be unknown
deployment.model_health

# Check service health. Should currently be unknown
deployment.service_health

# Check accuracy health. Should currently be unavailable
deployment.accuracy_health

# Check model deployed
deployment.model

# Check deployment capabilities 'supports_drift_tracking': True, 'supports_model_replacement': True
deployment.capabilities

# Check prediction usage - should be 0
deployment.prediction_usage

# List all your deployments
dr.Deployment.list()

# Get Drift Tracking Settings {u'target_drift': {u'enabled': False}, u'feature_drift': {u'enabled': False}}
deployment.get_drift_tracking_settings()

# Update Drift settings
deployment.update_drift_tracking_settings(target_drift_enabled=True, feature_drift_enabled=True)

# Get updated Drift Tracking Settings {u'target_drift': {u'enabled': True}, u'feature_drift': {u'enabled': True}}
deployment.get_drift_tracking_settings()

# Make predictions on scoring data against deployment id
response = requests.get(scoring_url)
data = response.text
predictions = make_datarobot_deployment_predictions(data, deployment.id)

# Refresh Deployment
deployment = dr.Deployment.get(deployment.id)

# Check model health. Should currently be failing as there's Drift (confirmed by UI)
deployment.model_health

# Check service health. Should currently be passing
deployment.service_health

# Check prediction usage - should be 1
deployment.prediction_usage

# Get new model
new_model_id = project.get_models()[0].id

# Replace model
deployment.replace_model(new_model_id, MODEL_REPLACEMENT_REASON.ACCURACY)

# Refresh Deployment
deployment = dr.Deployment.get(deployment.id)

# Check new model
deployment.model

# Check prediction usage - should be 0
deployment.prediction_usage

# Make predictions on scoring data against deployment id one replacement model
response = requests.get(scoring_url)
data = response.text
predictions = make_datarobot_deployment_predictions(data, deployment.id)

# Refresh Deployment
deployment = dr.Deployment.get(deployment.id)

# Check model health. Should currently be failing as there's Drift (confirmed by UI)
deployment.model_health

# Check service health. Should currently be passing
deployment.service_health

# Check prediction usage - should be 1
deployment.prediction_usage

# Delete Deployment
deployment.delete()

# List all your deployments to check it's gone
dr.Deployment.list()








