import os
import pandas as pd
import subprocess as sp

DATA_PATH=''
RAW_FILE='10K_Lending_Club_Loans.csv'
OUTPUT_FILE = 'results.csv'
FINAL_OUTPUT = 'predictionswithsource.csv'
currentDirectory = os.getcwd()
print(currentDirectory)
os.chdir(DATA_PATH)
print(DATA_PATH)


URL ='https://datarobot-support.orm.datarobot.com'
USERNAME = 'felix.huthmacher@datarobot.com'
API_TOKEN = 'XXXX'
DEPLOYMENT_ID = '5dc62756397e660e7c39acf4'

sourcedata = pd.read_csv(os.path.join(DATA_PATH, RAW_FILE))

command = ['batch_scoring_deployment_aware',
           '--host', URL,
           '--user', USERNAME,
           '--api_token', API_TOKEN,
           '--n_retry','5',
           '--pred_name','prediction',
           '--datarobot_key', DATAROBOT_KEY,
           '--no_verify_ssl',
           '--no-resume',
           '--out', OUTPUT_FILE,
           DEPLOYMENT_ID , RAW_FILE]

# echo command to python console
print(' '.join(command))

# execute - this will create a new file in the current directory
# ensure that this file does not exist prior to running, otherwise it will not continue the run
output = sp.check_output(command, stderr=sp.STDOUT, cwd=DATA_PATH)

# decode the batch scoring process output and dump to console
print(output.decode())
