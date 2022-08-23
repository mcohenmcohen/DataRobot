# Run this line in R to set the correct python version for Rstudio
# reticulate::use_python('/usr/local/bin/python3')

# Load Libraries
import datarobot
import pandas as pd
import numpy as np
import snowflake.connector
import os.path
import requests

# Pull the full project/blueprint table for R2 and save it
snowflake_connection = snowflake.connector.connect(
  user='zach',
  account='oh99766.us-east-1',
  authenticator='externalbrowser')
query = """
WITH project_blueprint_summary_raw AS (
  SELECT
    PROJECT_ID, BLUEPRINT_ID,
    MAX(ALL_METRICS_VALID_SCORE:"R Squared") AS R2
  FROM analytics.wall_e.STG_DR_APP_APP_LEADERBOARD
  WHERE
    TRY_TO_NUMBER(ALL_METRICS_VALID_SCORE:"R Squared"::text) IS NOT NULL AND
    ALL_METRICS_VALID_SCORE:"R Squared" != 0 AND
    PARTITION_INFO[0] != '(-1, -1)' AND
    IS_BLENDER = 0 AND
    CREATION_TIME between '2021-07-20 10:00:00' and  '2021-08-20 10:00:00'
  GROUP BY PROJECT_ID, BLUEPRINT_ID
)
SELECT * FROM project_blueprint_summary_raw
ORDER BY PROJECT_ID DESC, BLUEPRINT_ID DESC, R2 DESC
"""
# project_blueprint_summary_raw = pd.read_sql(query, snowflake_connection)
# project_blueprint_summary_raw.to_csv('/Users/zachary/Downloads/pid_pid_summary.csv', index=False)

# Set some parameters
# TODO: can we infer the correct metric based on the project?
QUERY_PID = '60b672e960d41a9c2ec9ff69'
METRIC = 'FVE Binomial' # R Squared, FVE Binomial, FVE Multinomial
MAX_SAMPLE_SIZE = 65
QUERY_SET = 'validation' # validation, crossValidation

# Pull leaderboard from public API
project = datarobot.Project.get(QUERY_PID)
all_models = project.get_models()

# Format leaderboard data for public api
# TODO: EXCLUDE INFINITIES
# TODO: EXCLUDE ZEROS
# TODO: EXCLUDE BLENDERS
# TODO: EXCLUDE MODELS >= THE VALIDATION MAX
QUERY_NORM = 0
TEMPLATE = "('{BID}', {METRIC})"
data = []
for model in all_models:
  metric_value = model.metrics[METRIC][QUERY_SET]
  if metric_value:
    metric_value = np.arcsinh(metric_value)
    QUERY_NORM += metric_value ** 2
    data.append(
      TEMPLATE.format(
        BID = model.blueprint_id,
        METRIC = metric_value
      ))
PID_BID_METRIC_DATA = ",\n".join(data)
QUERY_NORM = np.sqrt(QUERY_NORM)

# Load query template
with open(os.path.expanduser('~/workspace/data-science-scripts/zach/query_template.sql'),'r') as file:
    query_template = file.read()

query_template = query_template.format(
  QUERY_PID=QUERY_PID,
  METRIC=METRIC,
  PID_BID_METRIC_DATA=PID_BID_METRIC_DATA,
  QUERY_NORM=QUERY_NORM)

# Run query in snowflake to generate recs
recs = pd.read_sql(query_template, snowflake_connection)
snowflake_connection.close()

# Show results
i = 0
print(recs[['BLUEPRINT']].values[i][0])
print(recs.iloc[i,:])

# Connect to DR via the API and run the top 5 recs
# https://datarobot.slack.com/archives/C0286THNSSV/p1631627270025800?thread_ts=1631626103.023000&cid=C0286THNSSV
# staging:
# pid = "6140a32f264e73525eb8543b"
# host = "https://staging.datarobot.com/api/v2"
# AUTH = "API TOKEN"
#
# headers = {"Content-Type": "application/json", "Authorization": AUTH}
# s = requests.Session()
# endpoint = host + f"/projects/{}/blueprints/fromJson/".format(pid)
#
# res = s.post(
#     endpoint,
#     data=json.dumps({"modelType": "my custom blueprint", "blueprint": json.dumps(bp)}),
#     headers=headers,
# )
#
# print(res)
# print(res.content)
