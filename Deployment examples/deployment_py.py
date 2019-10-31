from bson import ObjectId

import datetime
import math
import requests
import time
import dateutil.parser as p
import pandas as pd

from mmm_performance.http import api
from mmm_performance.utils import date_to_isostr


class DeploymentApi(object):
    def __init__(self, app_host, pred_host, username, api_token, mongo_db, dr_key=None, instance_id=None,
                 is_aggregated=None):
        self.app_host = app_host + '/modelDeployments'
        self.pred_host = pred_host
        self.username = username
        self.api_token = api_token
        self.mongo_db = mongo_db
        self.dr_key = dr_key
        self.instance_id = instance_id
        self.is_aggregated = is_aggregated or False
        self.public_api = api.AuthenticatedRequests(api_token, app_host)

    @classmethod
    def from_environment(cls, env, use_aggregations=None):
        username = env.aggregate_user if use_aggregations else env.datarobot_user
        token = env.aggregate_api_token if use_aggregations else env.datarobot_api_token
        return cls(
            env.datarobot_endpoint,
            env.prediction_endpoint,
            username,
            token,
            env.get_mongo_db(),
            env.prediction_key,
            env.prediction_instance_id,
            is_aggregated=use_aggregations,
        )

    def _headers(self):
        headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json;charset=UTF-8',
            'Authorization': 'Token {}'.format(self.api_token)
        }
        return headers

    def create(self, pid, lid, label=None, description=None, wait=True):
        # Deploy the model
        deploy_body = {
            'projectId': pid,
            'modelId': lid,
            'label': label or 'Model {} from project {}'.format(lid, pid),
            'description': description,
        }
        if self.instance_id:
            deploy_body['instanceId'] = self.instance_id
        print("creating deployment with params {}".format(deploy_body))
        deploy_response = self.public_api.post('modelDeployments/asyncCreate',  body=deploy_body)
        location = deploy_response.headers['Location']
        if wait:
            print("waiting for deployment creation...")
            self.wait_for_job(location)
            print("deployment creation complete")
        return deploy_response.json()['id'], location

    def update_created_date(self, dep_id, created_date):
        # Since there's no way to do this via the API, we go straight to MongoDB
        collection = self.mongo_db.model_deployments
        query = {'_id': ObjectId(dep_id)}
        values = {'created_at': created_date,
                  'model_history.0.start_date': created_date,
                  'models.0.date': created_date,
                  'updated_at': created_date,
                  }
        collection.update_one(query, {'$set': values})

    def wait_for_job(self, async_location, max_wait=120):
        start_time = time.time()
        while time.time() < start_time + max_wait:
            response = requests.get(
                async_location, headers=self._headers(), allow_redirects=False)
            if response.status_code == 303:
                return response.headers['Location']
            assert response.status_code == 200
            data = response.json()
            if data['status'].lower()[:5] in ['error', 'abort']:
                print("Async task failed: {}".format(data))
                break
            print(".")
            time.sleep(5)
        raise RuntimeError('Client timed out waiting for {} to resolve'.format(async_location))

    def _list_deployments(self):
        return self.public_api.get('/modelDeployments', params={'skipStats': True}).json()

    def list(self):
        content = self._list_deployments()
        deployment_ids = [deployment['id'] for deployment in content['data']]
        return deployment_ids

    def list_deployed_models(self, pid=None):
        content = self._list_deployments()
        model_ids = []
        for deployment in content['data']:
            model_id = deployment['modelHistory'][0]['model']['id']
            project_id = deployment['modelHistory'][0]['project']['id']
            if project_id == pid:
                model_ids.append(model_id)
        return model_ids

    def get(self, dep_id, update_interval=None):
        update_interval = update_interval or 900
        response = self.public_api.get('/modelDeployments/{}'.format(dep_id),
                                       params={'updateInterval': update_interval})
        content = response.json()
        return content

    def replace(self, did, lid):
        body = {
            'modelId': lid,
            'reason': 'Data Drift'
        }
        return self.public_api.patch('modelDeployments/{}/model/'.format(did), body=body)

    def share(self, did, username, role='USER'):
        data = [{'username': username, 'role': role}]
        url = "modelDeployments/{}/accessControl/".format(did)
        return self.public_api.patch(url, body={'data': data})

    def predict(self, did, data, ts_spoof=None):
        headers = {
            'Content-Type': 'application/json; charset=UTF-8',
        }
        if self.dr_key:
            headers['datarobot-key'] = self.dr_key
        # NOTE user needs experimental API access :(
        if ts_spoof:
            headers['X-DataRobot-Prediction-Timestamp'] = ts_spoof
        predictions_response = requests.post(
            self.pred_host + '/deployments/{}/predictions'.format(did),
            auth=(self.username, self.api_token), data=data, headers=headers)

        predictions_response.raise_for_status()
        return predictions_response

    def make_predictions_from_scenario(self, did, scenario, start=None):
        dep = self.get(did)
        if not start:
            start = p.parse(dep['createdAt'])
        df = pd.read_csv(scenario.scoring_path)
        pred_per_hour = scenario.total_predictions / float(scenario.num_hours * scenario.snapshots)
        pred_per_hour = int(math.ceil(pred_per_hour))
        print("predicting {} hours into future {} rows per hour".format(
            scenario.num_hours, pred_per_hour))
        total_prediction_rows = 0
        for k in range(scenario.num_hours):
            pred_date = start + datetime.timedelta(hours=k)
            pred_ts = pred_date.strftime('%Y-%m-%dT%H:%M:%S.%f')
            num_predictions = 0
            while num_predictions < pred_per_hour:
                data = df.sample(pred_per_hour, replace=True)
                try:
                    self.predict(did, data.to_json(orient='records'), ts_spoof=pred_ts)
                    total_prediction_rows += len(data)
                except Exception as e:
                    print("Prediction ERROR: dep={} predts={} exception={}".format(
                        did, pred_ts, str(e)))
                finally:
                    num_predictions += len(data)
        print("deployment = {} num_hours = {} num_rows = {}".format(
            did, scenario.num_hours, total_prediction_rows))
        end = pred_date + datetime.timedelta(seconds=3600)
        return total_prediction_rows, start, end

    def start_date(self, did, start=None):
        dep = self.get(did)
        if not start:
            start = p.parse(dep['createdAt'])
        return start

    def end_date(self, did, scenario, start=None):
        return self.start_date(did, start) + datetime.timedelta(hours=scenario.num_hours)

    def run_deployment_response_stats_suite(self, did, lid, scenario, start_date=None,
                                            end_date=None, snapshot=None, deployment_count=None,
                                            full_interval_only=None):
        """Get stats for a deployment for different time periods"""
        start_date = start_date or self.start_date(did)
        end_date = end_date or self.end_date(did, scenario)
        records = []
        hours_interval = 12
        end = False
        interval_end = end_date if full_interval_only else start_date
        while True:
            interval_end = interval_end + datetime.timedelta(hours=hours_interval)
            if interval_end >= end_date:
                end = True
                interval_end = end_date

            records.extend(self.get_deployment_metric_response_stats(
                did, lid, scenario, start_date, interval_end, snapshot, deployment_count))
            if end:
                pred_count = records[-1]['prediction_count']
                num_hours = int((end_date - start_date).total_seconds() // 3600)
                records.extend([
                    self.get_dep_response_time(did, pred_count, num_hours, snapshot,
                                               deployment_count),
                    self.list_deps_response_time(did, pred_count, num_hours, snapshot,
                                                 deployment_count)
                ])
                break
        return records

    def _build_stats_result(self, did, snapshot, num_hours, prediction_count, endpoint, response,
                            deployment_count):
        return {
            'deployment_id': did,
            'snapshot': snapshot,
            'num_hours': num_hours,
            'prediction_count': prediction_count,
            'endpoint': endpoint,
            'response_time_seconds': response.elapsed.total_seconds() if response else None,
            'is_aggregated': self.is_aggregated,
            'deployment_count': deployment_count,
            'series_name': ('aggregated ' if self.is_aggregated else '') + endpoint
        }

    def get_deployment_metric_response_stats(self, did, lid, scenario, start_date, end_date,
                                             snapshot, deployment_count):
        """Response times for fetching different deployment related endpoints"""
        responses = {
            'serviceStats': self.fetch_service_stats(did, scenario, start_date, end_date),
            'dataDrift': self.fetch_feature_drift_stats(did, lid, scenario, start_date, end_date),
            'targetDrift': self.fetch_target_drift_stats(did, lid, scenario, start_date, end_date),
        }
        stats_data = responses['serviceStats'].json()
        num_hours = int((end_date - start_date).total_seconds() // 3600)
        prediction_count = stats_data['totalPredictionRows']
        for endpoint, response in responses.items():
            yield self._build_stats_result(did, snapshot, num_hours, prediction_count, endpoint,
                                           response, deployment_count)

    def list_deps_response_time(self, did, prediction_count, num_hours, snapshot, deployment_count):
        response = self.public_api.get('modelDeployments/',
                                       params={'skipStats': False, 'updateInterval': 0})
        endpoint = 'modelDeployments'
        return self._build_stats_result(did, snapshot, num_hours, prediction_count, endpoint,
                                        response, deployment_count)

    def get_dep_response_time(self, did, prediction_count, num_hours, snapshot, deployment_count):
        response = self.public_api.get('modelDeployments/{}'.format(did),
                                       params={'skipStats': False, 'updateInterval': 0})
        endpoint = 'modelDeployments/:id'
        return self._build_stats_result(did, snapshot, num_hours, prediction_count, endpoint,
                                        response, deployment_count)

    def fetch_service_stats(self, did, scenario, start_date=None, end_date=None):
        start_date = start_date or self.start_date(did)
        end_date = end_date or self.end_date(did, scenario)
        url = 'modelDeployments/{did}/serviceStats/'.format(did=did)
        return self.public_api.get(url, params={'start': date_to_isostr(start_date),
                                   'end': date_to_isostr(end_date)})

    def fetch_feature_drift_stats(self, did, lid, scenario, start_date=None, end_date=None):
        start_date = start_date or self.start_date(did)
        end_date = end_date or self.end_date(did, scenario)
        url = 'modelDeployments/{did}/dataDrift/'.format(did=did)
        return self.public_api.get(url, params={'start': date_to_isostr(start_date),
                                                'end': date_to_isostr(end_date),
                                                'compareWith': 'training',
                                                'updateInterval': 0,
                                                'modelId': lid})

    def fetch_target_drift_stats(self, did, lid, scenario, start_date=None, end_date=None):
        start_date = start_date or self.start_date(did)
        end_date = end_date or self.end_date(did, scenario)

        if (end_date - start_date) < datetime.timedelta(days=1):
            return None

        url = 'modelDeployments/{did}/targetDrift/'.format(did=did)
        return self.public_api.get(url, params={'start': date_to_isostr(start_date),
                                                'end': date_to_isostr(end_date),
                                                'modelId': lid})
