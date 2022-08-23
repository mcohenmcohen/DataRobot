# This sample code shows how you can create a project with data from the AI catalog, 
# and then subsequently get scores for the training data as well as additional scoring data

import os
import csv
import pandas as pd
import datarobot as dr
from datetime import date
import requests
import json
import time
import io

dataframe_collection = {}
def createDFarray(df, size_threshold, part_number=0):
    # play around with size and threshold to get the right size, ultimately it needs to be less than 1GB
    file_size = df.memory_usage(index=True, deep=False).sum() / 1038336
    print(file_size)
    num_records = len(df)

    if file_size > size_threshold:
        records_to_split_off = int(num_records * size_threshold // file_size)
        df_to_save = df.head(records_to_split_off)
        dataframe_collection[part_number] = pd.DataFrame(df_to_save)
        createDFarray(df.tail(num_records-records_to_split_off), size_threshold, part_number=part_number+1)

    else:
        dataframe_collection[part_number] = pd.DataFrame(df)
        return dataframe_collection

def wait_for_async_resolution(client, status_url):
        status = False
        while status == False:
            resp = client.get(status_url)
            r = json.loads(resp.content)
            try:
                statusjob = r['status'].upper()
                
            except:
                statusjob = ''
            if resp.status_code == 200 and statusjob != 'RUNNING' and statusjob != 'INITIALIZED': 
                status = True
                return resp
            time.sleep(10)   # Delays for 10 seconds.

def wait_for_result(client, response):
    assert response.status_code in (200, 201, 202), response.content

    if response.status_code == 200:
        data = response.json()

    elif response.status_code == 201:
        status_url = response.headers['Location']
        resp = client.get(status_url)
        assert resp.status_code == 200, resp.content
        data = resp.json()

    elif response.status_code == 202:
        status_url = response.headers['Location']
        resp = wait_for_async_resolution(client, status_url)
        data = resp.json()
    return data

# trainingdata file
# https://dataset-for-performance-testing.s3.amazonaws.com/airlines_5gb.csv
FILENAME = '../SampleData/airlines_5gb.csv'

# setup your headers for the calls to the app and prediction server
API_TOKEN = '<YOUR TOKEN>'
USERNAME = 'felix.huthmacher@datarobot.com'
MODELMANAGEMENTENDPOINT = 'https://app.datarobot.com/api/v2'
MODELMANAGEMENTHEADERS = {'Authorization': 'token %s' % API_TOKEN }
DATAMANAGEMENTHEADERS = {'Content-Type': 'application/json; charset=UTF-8', 'Authorization': 'token %s' % API_TOKEN }

drclient = dr.Client(endpoint=MODELMANAGEMENTENDPOINT, token=API_TOKEN,connect_timeout=1800)


TARGET = 'ArrDelay'
PROJECTNAME = 'DataSetScoringDemo '

USE_AUTOPILOT = True
USE_EXISTING_MODEL = False

# 1. upload training data to catalog
if (USE_EXISTING_MODEL == False):

    #response = drclient.build_request_with_file("POST",
    #                                           "%s/datasets/fromFile/" % (MODELMANAGEMENTENDPOINT),
    #                                           "DataSetScoringDemo ",
    #                                           file_path=FILENAME,
    #                                           read_timeout=1200
    #                                          )
    response = requests.post("%s/datasets/fromFile/" % (MODELMANAGEMENTENDPOINT),
                            headers=MODELMANAGEMENTHEADERS,  files = payload, timeout = 1800 )
   

    # wait till file is uploaded to catalog
    datacatalog_response = wait_for_result(drclient, response)

    # get dataset details
    dataset_response = requests.get("%s/datasets/%s/" % (MODELMANAGEMENTENDPOINT, datacatalog_response['datasetId']),
                            headers=DATAMANAGEMENTHEADERS)
    dataset = dataset_response.json()

# 2. create a new project with data from catalog
    payload = {
        'projectName': 'AI Catalog Demo Project ' + str(date.today()),
        'datasetId': str(dataset['datasetId']),
        'datasetVersionId': str(dataset['versionId']),
        'user': USERNAME,
        'password': API_TOKEN
    }
    project_response = drclient.post(
        '/projects/',
        data=payload,
        headers={'Content-Type': 'application/json'}
    )    

    print(project_response)
    projectID = project_response.json()['pid']
    # wait till project is created
    project_response = wait_for_result(drclient, project_response)



# 2a. Creates new featurelist for each project as needed

if (USE_EXISTING_MODEL == False):
    newProject = dr.Project(projectID)
    feature_names = []
    # get all features from dataset
    # Substracts lists by converting them into sets (order not preserved)
    for i in newProject.get_features():
        feature_names.append(i.name)
    
    def list_diff(li1, li2):
        return (list(set(li1) - set(li2)))

    # 2b. create new featurelist by removing the below features from the featurelist
    unwanted_features = ['TaxiIn', 'TaxiOut', 'TailNum']
    project_featurelist = list_diff(feature_names, unwanted_features)
    newFeatureList = newProject.create_featurelist("featurelist_500_01", project_featurelist)
    
    
if (USE_AUTOPILOT == True) & (USE_EXISTING_MODEL == False):
    newProject.set_target(target=TARGET,
                       mode=dr.AUTOPILOT_MODE.QUICK,
                       worker_count= 8,
                       featurelist_id=newFeatureList.id,
                       max_wait= 36000000
                       )
    newProject.wait_for_autopilot()
    recommendation_type = dr.enums.RECOMMENDED_MODEL_TYPE.RECOMMENDED_FOR_DEPLOYMENT
    recommendation = dr.models.ModelRecommendation.get(newProject.id, recommendation_type)
    bestModelId = recommendation.model_id
    newProjectId = newProject.id

if (USE_AUTOPILOT == False) & (USE_EXISTING_MODEL == False):
    newProject.set_target(target=TARGET,
                       mode=dr.AUTOPILOT_MODE.MANUAL,
                       worker_count= 8,
                       featurelist_id=newFeatureList.id,
                       max_wait= 36000000
                       )

    # pick any blueprint/model from repository
    blueprints = newProject.get_blueprints()
    for blueprint in blueprints:
        if blueprint.model_type == 'RuleFit Regressor':
            bestblueprint = blueprint
            break

    JobId = newProject.train(bestblueprint, sample_pct=50)
    newModel = dr.models.modeljob.wait_for_async_model_creation(project_id=newProject.id, model_job_id=JobId)
    fi = newModel.get_or_request_feature_impact(600)
    newModel.cross_validate()
    bestModelId = newModel.id
    newProjectId = newProject.id

if (USE_EXISTING_MODEL == True):
    newProjectId = '5dc56e96ef50aa01946fde3d' 
    bestModelId = '5dc56f03ef50aa01c76fdedb'
    dataset = json.loads('{ "datasetId":"5dc4e1e7397e66791c9b3375", "versionId":"5dc4e1e8397e66791c9b3376"}')
    print(dataset['datasetId'])


# 3. score training data with a model from the project
model = dr.Model.get(model_id=bestModelId,project=newProjectId)
pred_job = model.request_training_predictions(dr.enums.DATA_SUBSET.ALL)
MAX_WAIT = 60 * 60  # Maximum number of seconds to wait for prediction job to finish
predictions = pred_job.get_result_when_complete(max_wait=MAX_WAIT)
predictions.to_csv('trainingdata_results.csv')

# 4. score additional data with a model from the project

# check size of scoringdata
# each <1GB dataframe for prediction data

# just for demo purposes a subset of the data
# <1GB for scoring data upload to DR
scoringdata = pd.read_csv(FILENAME,nrows=10000,encoding="ISO-8859-1")

dataframe_collection = createDFarray(scoringdata, 0.1, part_number=0)

for key in dataframe_collection.keys():
    
    csv_string = dataframe_collection[key].to_csv()
    csv_string = str.encode(csv_string)
    payload = {
       'description': 'testCatalogScoring ' + str(key),
       'file': (io.BytesIO(csv_string))
    }
    #response = drclient.build_request_with_file("POST",
    #                                           "%s/datasets/fromFile/" % (MODELMANAGEMENTENDPOINT),
    #                                           "testCatalogScoring ",
    #                                           file_path=FILENAME,
    #                                           read_timeout=1200
    #                                          )
    response = requests.post("%s/datasets/fromFile/" % (MODELMANAGEMENTENDPOINT),
                            headers=MODELMANAGEMENTHEADERS,  files = payload, timeout = 1800 )

    # wait till file is uploaded to catalog
    datacatalog_response = wait_for_result(drclient, response)

    # get dataset details
    dataset_response = requests.get("%s/datasets/%s/" % (MODELMANAGEMENTENDPOINT, datacatalog_response['datasetId']),
                            headers=MODELMANAGEMENTHEADERS)
    dataset = dataset_response.json()
    
    # Retrieves the details of the dataset with given ID and version ID
    # /api/v2/datasets/(datasetId)/versions/(datasetVersionId)/
    #dataset_response = requests.get("%s/datasets/%s/versions/%s/" % (MODELMANAGEMENTENDPOINT, dataset['datasetId'],dataset['versionId']),
    #                        headers=MODELMANAGEMENTHEADERS)
    #dataset = dataset_response.json()

    # link dataset to project & model
    payload = {
        'datasetId': str(dataset['datasetId']),
        'datasetVersionId': str(dataset['versionId']) 
    }

    response_assoc = drclient.post(
        '/projects/%s/predictionDatasets/datasetUploads/' % (newProjectId),
        data=payload,
        headers={'Content-Type': 'application/json'}
    )                          
    predictiondata = response_assoc.json()

    # score data
    model = dr.Model.get(model_id=bestModelId,project=newProjectId)
    pred_job = model.request_predictions(predictiondata['datasetId'])
    MAX_WAIT = 60 * 60  # Maximum number of seconds to wait for prediction job to finish
    predictions = pred_job.get_result_when_complete(max_wait=MAX_WAIT)
    predictions.to_csv('scoringdata_results.csv')
    for row in predictions.iterrows():
        print(row)
