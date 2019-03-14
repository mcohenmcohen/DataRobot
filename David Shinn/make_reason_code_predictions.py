import sys

import pandas as pd
import requests

PREDICTION_FILE = 'pred.csv'
OUTPUT_FILE = 'reason_codes_long_format.csv'

PROJECT_ID = '5ab1519e5feaa758b227f436'
MODEL_ID = '5ab3c9ba5feaa7572313d02f'
MAX_CODES = 10
POSITIVE_CLASS = 1
BATCH_SIZE = 100 # how many records to send each time, typically less than 100 rows

USERNAME = ''
API_TOKEN = ''
SERVER_URL = ''
SERVER_KEY = ''

headers = {'Content-Type': 'text/plain; charset=UTF-8', 'datarobot-key': SERVER_KEY}

url_request = "{server_url}/predApi/v1.0/{project_id}/{model_id}/reasonCodesPredictions".format(
        server_url=SERVER_URL, project_id=PROJECT_ID, model_id=MODEL_ID)

def mini_batch_file(filepath, batch_size, encoding='utf-8'):
    with open(filepath) as infile:
        header = infile.readline()
        output_lines = []
        for line in infile:
#             output_lines.append(line.decode(encoding))  # May be needed if python 2
            output_lines.append(line.decode)
            if len(output_lines) == batch_size:
                output_lines.insert(0, header) # put header up front
                yield ''.join(output_lines) # output csv text with header
                output_lines = []
        else:
            output_lines.insert(0, header) # put header up front
            yield ''.join(output_lines) # output csv text with header

def get_probability_prediction(rc_json_row, positive_class):
    return [p for p in rc_json_row['predictionValues'] if p['label'] == positive_class][0]['value']

params = {'maxCodes': MAX_CODES}

reason_code_lines = [] # long so will have MAX_CODES * num_record lines
for batch_num, batch in enumerate(mini_batch_file(PREDICTION_FILE, BATCH_SIZE)):
    try:
        sys.stderr.write('\r--- Making request {:,} totalling {:,} rows requested'.format(
            batch_num+1, (batch_num+1)*BATCH_SIZE))

        row_id_offset = batch_num*BATCH_SIZE # needed so row_id reflected of prediction file
        data = batch.encode('utf-8') # encoding must match in headers
        predictions_response = requests.post(url_request,
                                            auth=(USERNAME, API_TOKEN),
                                            data=data,
                                            headers=headers,
                                            params=params,
                                            timeout=120)
        if predictions_response.status_code != 200:
            try:
                message = predictions_response.json().get('message', predictions_response.text)
                status_code = predictions_response.status_code
                reason = predictions_response.reason

                print(u'Status: {status_code} {reason}. Message: {message}.'.format(message=message,
                                                                                    status_code=status_code,
                                                                                    reason=reason))
            except ValueError:
                print('Prediction failed: {}'.format(predictions_response.reason))
                predictions_response.raise_for_status()
        else:
            reason_code_prediction_rows = predictions_response.json()['data']
            for rc_json_row in reason_code_prediction_rows:
                prediction = get_probability_prediction(rc_json_row, POSITIVE_CLASS)
                row_id = rc_json_row['rowId'] + row_id_offset
                for rc in rc_json_row['reasonCodes']:
                    rc['prediction'] = prediction
                    rc['row_id'] = row_id
                    reason_code_lines.append(rc)
    except KeyboardInterrupt:
        break

pd.DataFrame(reason_code_lines).to_csv(OUTPUT_FILE, index=False)
