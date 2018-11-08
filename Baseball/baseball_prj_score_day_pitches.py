import pandas as pd
import datarobot as dr
import mlb_pull_year as mlb
import requests
import os
import time
from pprint import pprint

API_TOKEN = os.getenv('DATAROBOT_API_TOKEN')
ENDPOINT = os.getenv('DATAROBOT_ENDPOINT')
dr.Client(endpoint=ENDPOINT, token=API_TOKEN)

PROJECT_ID = '5bdb7caa7c6f8b71e0428016'  # The bseball project
project = dr.Project.get(PROJECT_ID)

DEPLOYMENT_ID = '5bdf672f7c6f8b2939428077'  # Eg, the project's recommended model: XGBoost @ 80%
USERNAME = os.getenv('DATAROBOT_USERNAME')

TRAINING_DATA = 'pitch_scoring.csv'  # Only used if you train a new baseball project


def create_new_baseball_project():
    '''
    Helper function to create a new baseball project using source training data
    '''
    t1 = time.time()
    # Read source data
    pitches_train = pd.read_csv(TRAINING_DATA, parse_dates=['date'])
    print('Source data shape:', pitches_train.shape)
    # pitches_train.head()

    # Create the project in the DataRobot Cloud
    print('Creating project')
    project = dr.Project.create(sourcedata=pitches_train, project_name='Baseball pitch prediction')

    # Set target starts autopilog
    print('Running autopilot')
    project.set_target(target='strike', mode='auto', worker_count=20)

    # Block until complete
    print('Waiting to complete')
    project.wait_for_autopilot()

    print('- Autopilot done. Time: %.3f' % (time.time() - t1))

    return project


def get_day_pitches(year, month, day):
    '''
    Retrieve pitch data for a day from mlb.com
    '''
    pitches = mlb.read_yearmonth(year, month, day)  # omits the 'strike' feature
    print('\nNum pitches:', len(pitches))

    #
    # Edit the columns for the received day's pitches to match the training data columns
    #

    # Get features from the daily pitch data
    pitches_today = pd.DataFrame(pitches)
    all_pitch_cols = pitches_today.columns.sort_values().tolist()
    print("day's pitch data:", pitches_today.shape)

    # Get features from the project
    # cols_train = pitches_train.columns.sort_values()
    fl = [fl for fl in project.get_featurelists() if fl.name == 'Raw Features'][0]
    cols_train = fl.features

    # Prune the daily pitch data features to match the project features
    cols_to_drop = [feat for feat in all_pitch_cols if feat not in cols_train]
    pitches_today = pitches_today.drop(cols_to_drop, axis=1)
    cols_pred = pitches_today.columns.tolist()

    print('pitches_pred columns len:', len(cols_pred))
    print('pitches_train columns len:', len(cols_train))

    return pitches_today


#
# Score the day's pitch data on the deployment
#

# Need to write df to file then read back in to get the request.post to work.  Not ideal.
pred_file = 'pitch_pred.csv'
pitches_today.to_csv(pred_file)
data = open(pred_file, 'rb').read()  # This works.  This is type bytes: print(type(data))
# print(data)
print('pred file shape:', pitches_today.shape)

headers = {'Content-Type': 'text/plain; charset=UTF-8', 'datarobot-key': '544ec55f-61bf-f6ee-0caf-15c7f919a45d'}
predictions_response = requests.post('https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/%s/predictions' % (DEPLOYMENT_ID),
                                     auth=(USERNAME, API_TOKEN), data=data, headers=headers)

predictions_response.raise_for_status()
df = pd.DataFrame(predictions_response.json().get('data'))

# Flatten nested label/value dict via apply
df['label1'] = None
df['proba1'] = None
df['label2'] = None
df['proba2'] = None
def func(row):
    for i, pair in enumerate(row['predictionValues']):
        name = pair.get('label')
        val = pair.get('value')
        col_name_n = 'label' + str(i+1)
        col_val_n = 'value' + str(i+1)
        row[col_name_n] = name
        row[col_val_n] = val
    return row
df_pred_out = df.apply(lambda row: func(row), axis=1)
del df_pred_out['predictionValues']
print('pred out file shape:',df_pred_out.shape)

df_pred_out.head()

if __name__ == '__main__':
    # project = create_new_baseball_project()

    pitches_today = get_day_pitches(2018, 4, 1)
    pitches_today.shape
