import datetime
import os
import requests
import time

from dateutil import parser
import random
import numpy as np
import pandas as pd

# can grab all this from the Integrations tab
# TODO move to config file or add cli
API_TOKEN =  '-tthMRg4-sXF21D7FTQYxfUAMZSGWqvs' 
USERNAME = 'aengel@datarobot.com' 
DATAROBOT_KEY = '544ec55f-61bf-f6ee-0caf-15c7f919a45d'  
headers = {'Content-Type': 'application/json', 'datarobot-key': DATAROBOT_KEY}


CHAMPION_DEPLOYMENT_ID = '5bad2050e56fab405ccf6017'
# NOTE add a challenger deployment ID to route ~20% of requests to
CHALLENGER_DEPLOYMENT_ID = None

# prepare the data
fname = 'pitch_scoring.csv'
pred_file = os.path.join(os.getcwd(), fname)
target = 'strike'
dt = 'date'

df = pd.read_csv(pred_file)
df = df.drop(target, axis=1)
# turn the timestamp column into dates, then break into segments based on total
# days and desired cycles
df['date_segments'] = df[dt].apply(parser.parse)
df = df.sort('date_segments')
s_date = df.date_segments.min()
e_date = df.date_segments.max()
#num_days = (e_date - s_date).days
#cycles = 14
#days_per_cycle = num_days / cycles
days_per_cycle = 7
periods = {}
for c in range(cycles):
    p_start = s_date + datetime.timedelta(days=days_per_cycle * c)
    p_end = p_start + datetime.timedelta(days=days_per_cycle)
    periods[c] = df[(df.date_segments >= p_start) & (df.date_segments <= p_end)]


MAX_ROWS = 5
DELAY_MIN = 8
DELAY_MAX = 12
# we want ~20k prediction rows a day which is ~14 per minute
# current avg will be be ~18 rows per minute


# NOTE change cycle_start to be an explict date if you stop + start the script but want to maintain the cycles.
# using utcnow() will start the cycle over again 
# start = datetime.datetime.utcnow()
start = datetime.datetime(2018, 9, 18)
end = start + datetime.timedelta(days=cycles)

now = datetime.datetime.utcnow()
num_req = 0
total_rows = 0
msg = "Starting prediction loop with settings...\n"
msg += "cycles: {} | days_per_cycle: {} | max_rows_per_req: {} | min_delay {} | max_delay {}".format(
    cycles, days_per_cycle, MAX_ROWS, DELAY_MIN, DELAY_MAX)
print(msg)

while now < end:
    curr_cycle = (now - start).days
    num_rows = random.randint(1, MAX_ROWS)
    total_rows += num_rows
    data = periods[curr_cycle]
    data = data.drop('date_segments', axis=1)
    # randomly sample some rows within this time periods
    pred_data = data.ix[random.sample(data.index, num_rows)]
    pred_data = pred_data.to_json(orient='records')
    if random.random() > 0.2 or CHALLENGER_DEPLOYMENT_ID is None:
        predictions_response = requests.post('https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/%s/predictions' % (CHAMPION_DEPLOYMENT_ID),
                                        auth=(USERNAME, API_TOKEN), data=pred_data, headers=headers)

        predictions_response.raise_for_status()

    else:
        predictions_response = requests.post('https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/%s/predictions' % (CHALLENGER_DEPLOYMENT_ID),
                                         auth=(USERNAME, API_TOKEN), data=pred_data, headers=headers)
        predictions_response.raise_for_status()
    num_req += 1
    if num_req % 5 == 0 or num_req == 1:
        print("Current cycle: {} | Total Reqs: {} | Total Rows: {}".format(curr_cycle, num_req,
            total_rows))

    sleep = random.uniform(DELAY_MIN, DELAY_MAX)
    time.sleep(sleep)
    now = datetime.datetime.utcnow()

