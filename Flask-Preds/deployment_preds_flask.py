import pandas as pd
import datarobot as dr
import os
import sys
import requests
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from pandas.io.json import json_normalize

USERNAME = os.environ['DATAROBOT_USERNAME']
API_KEY = os.environ['DATAROBOT_API_KEY']
DATAROBOT_KEY = os.environ['DATAROBOT_KEY']
MAX_PREDICTION_FILE_SIZE_BYTES = 52428800  # 50 MB

UPLOAD_FOLDER = './uploads'
DOWNLOAD_FOLDER = './downloads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

API_TOKEN = os.environ['DATAROBOT_API_TOKEN']
ENDPOINT = 'https://app.datarobot.com/api/v2'
dr.Client(token=API_TOKEN, endpoint=ENDPOINT)


class DataRobotPredictionError(Exception):
    """Raised if there are issues getting predictions from DataRobot"""
    print('*** Exception:', Exception)


def make_datarobot_deployment_predictions(data, deployment_id, forecast_point=None):
    """
    Make predictions on data provided using DataRobot deployment_id provided.
    See docs for details:
         https://app.datarobot.com/docs/users-guide/predictions/api/new-prediction-api.html

    Parameters
    ----------
    data : str
        Feature1,Feature2
        numeric_value,string
    deployment_id : str
        The ID of the deployment to make predictions with.

    Returns
    -------
    Response schema:
        https://app.datarobot.com/docs/users-guide/predictions/api/new-prediction-api.html#response-schema

    Raises
    ------
    DataRobotPredictionError if there are issues getting predictions from DataRobot
    """
    print('- IN make_datarobot_deployment_predictions')
    # Set HTTP headers. The charset should match the contents of the file.
    headers = {'Content-Type': 'text/plain; charset=UTF-8', 'datarobot-key': DATAROBOT_KEY}
    headers = {'Content-Type': 'application/json; charset=UTF-8', 'datarobot-key': '544ec55f-61bf-f6ee-0caf-15c7f919a45d'}
    # data = pd.read_csv(PRED_FILE)
    data = data.to_json(orient='records')

    url = 'https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/'\
          'predictions'.format(deployment_id=deployment_id)
    # Make API request for predictions
    print('- url:', url)
    predictions_response = requests.post(
        url, auth=(USERNAME, API_KEY), data=data, headers=headers)
    print('- predictions_response type:', type(predictions_response))
    _raise_dataroboterror_for_status(predictions_response)
    # Return a Python dict following the schema in the documentation
    print('- OUT make_datarobot_deployment_predictions')
    return predictions_response.json()


def _raise_dataroboterror_for_status(response):
    """Raise DataRobotPredictionError if the request fails along with the response returned"""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        err_msg = '{code} Error: {msg}'.format(
            code=response.status_code, msg=response.text)
        raise DataRobotPredictionError(err_msg)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_deployments():
    deployments = dr.Deployment.list()
    return deployments


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    deployments = get_deployments()

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        deployment_id = request.form['deployment_id']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename,
                                    deployment_id=deployment_id))

    return render_template('deployments.html', deployments=deployments)


@app.route('/uploads/<deployment_id>/<filename>')
def uploaded_file(filename, deployment_id):
    f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(f, encoding='iso-8859-1')
    data_size = sys.getsizeof(data)
    if data_size >= MAX_PREDICTION_FILE_SIZE_BYTES:
        print(
            'Input file is too large: {} bytes. '
            'Max allowed size is: {} bytes.'
        ).format(data_size, MAX_PREDICTION_FILE_SIZE_BYTES)
        return 1
    try:
        predictions = make_datarobot_deployment_predictions(data, deployment_id)
        print('- return from make_datarobot_deployment_predictions')
    except DataRobotPredictionError as exc:
        print(exc)
        return 1

    # Get preds as dataframe, write to csv, send from dir
    df_preds = json_normalize(predictions['data'])
    print(type(df_preds))
    values = df_preds['predictionValues'].values[0]
    for i, vals in enumerate(values):
        label_name = str(values[i].get('label'))
        df_preds[str('label_'+label_name)] = df_preds['predictionValues'].apply(lambda x: x[i].get('label'))
        df_preds[str('value_'+label_name)] = df_preds['predictionValues'].apply(lambda x: x[i].get('value'))
    df_preds.drop(['predictionValues'], axis=1, inplace=True)
    file_out = 'pred_out.csv'
    df_preds.to_csv(os.path.join(app.config['DOWNLOAD_FOLDER'], file_out), index=False)

    return send_from_directory(app.config['DOWNLOAD_FOLDER'], file_out)
