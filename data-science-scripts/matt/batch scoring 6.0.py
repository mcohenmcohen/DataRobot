"""
Usage:
    python datarobot-predict.py <input-file.csv> <output-file.csv>
 
This example uses the requests library which you can install with:
    pip install requests
We highly recommend that you update SSL certificates with:
    pip install -U urllib3[secure] certifi
 
Details: https://app.datarobot.com/docs/early-release/batch-prediction-api.html
"""
import argparse
import sys
import time
 
import requests
 
 
API_KEY = 'XXXX'
BATCH_PREDICTIONS_URL = 'https://app.datarobot.com/api/v2/batchPredictions'
DEPLOYMENT_ID = '5d88c990a57acd12dc16d3ef'
POLL_INTERVAL = 10
 
 
class DataRobotPredictionError(Exception):
    """Raised if there are issues getting predictions from DataRobot"""
 
 
class JobStatus(object):
    INITIALIZING = 'INITIALIZING'
    RUNNING = 'RUNNING'
    COMPLETED = 'COMPLETED'
    ABORTED = 'ABORTED'
 
 
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, usage='python %(prog)s <input-file.csv> <output-file.csv>'
    )
    parser.add_argument(
        'input_file', type=argparse.FileType('rb'), help='Input CSV file with data to be scored.'
    )
    parser.add_argument(
        'output_file', type=argparse.FileType('wb'), help='Output CSV file with the scored data.'
    )
    return parser.parse_args()
 
 
def main():
    args = parse_args()
 
    input_file = args.input_file
    output_file = args.output_file
    try:
        download_url = make_datarobot_batch_predictions(input_file)
    except DataRobotPredictionError as err:
        print('Error: {}'.format(err))
        return 1
    else:
        download_datarobot_batch_predictions(download_url, output_file)
        print('Results downloaded to: {}'.format(output_file.name))
        return 0
 
 
def make_datarobot_batch_predictions(input_file):
    # Create new job for batch predictions
    payload = {
        'deploymentId': DEPLOYMENT_ID,
        'maxExplanations': 5,
        'thresholdHigh': 0.5,
        'thresholdLow': 0.15,
    }
    job = _request('post', BATCH_PREDICTIONS_URL, json=payload)
    links = job['links']
 
    # Upload scoring data
    upload_url = links['csvUpload']
    headers = {'Content-Type': 'text/csv'}
    _request('put', upload_url, data=input_file, headers=headers, to_json=False)
 
    # Wait until job's complete
    job_url = links['self']
    while True:
        job = _request('get', job_url)
        status = job['status']
        if status in {JobStatus.INITIALIZING, JobStatus.RUNNING}:
            print('Waiting for the job to complete: {}%'.format(job['percentageCompleted']))
            time.sleep(POLL_INTERVAL)
            continue
        elif status == JobStatus.COMPLETED:
            return job['links']['download']
        raise DataRobotPredictionError(job['statusDetails'])
 
 
def download_datarobot_batch_predictions(download_url, output_file):
    response = _request('get', download_url, to_json=False)
    output_file.write(response)
 
 
def _request(method, url, data=None, json=None, headers=None, to_json=True):
    if not headers:
        headers = {}
    headers.setdefault('Authorization', 'Token {}'.format(API_KEY))
    response = getattr(requests, method)(url, headers=headers, data=data, json=json)
    _raise_datarobot_error_for_status(response)
    if to_json:
        return response.json()
    return response.content
 
 
def _raise_datarobot_error_for_status(response):
    """Raise DataRobotPredictionError if the request fails along with the response returned."""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        err_msg = '{code} Error: {msg}'.format(code=response.status_code, msg=response.text)
        raise DataRobotPredictionError(err_msg)
 
 
if __name__ == '__main__':
    sys.exit(main())
 