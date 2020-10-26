import datarobot as dr
import pandas as pd

from src.ts_projects import get_top_models_from_projects


###############
# Predictions
###############
def series_to_clusters(df, ts_settings, split_col='Cluster'):
    series_id = ts_settings['series_id']

    series = df[[series_id, split_col]].drop_duplicates().reset_index(drop=True)
    series_map = {k: str(v) for (k, v) in zip(series[series_id], series[split_col])}
    return series_map


def clusters_to_series(df, ts_settings, split_col='Cluster'):
    series_id = ts_settings['series_id']

    df = df[[series_id, split_col]].drop_duplicates().reset_index(drop=True)
    groups = df.groupby(split_col)[series_id].apply(lambda x: [i for i in x])

    clusters_to_series = {str(i + 1): g for i, g in enumerate(groups)}
    return clusters_to_series


def get_project_stats(
    projects, n_models, cluster_to_series_map, metric=None, split_col='Cluster', prefix='TS'
):
    stats = pd.DataFrame()
    for i, p in enumerate(projects):
        if metric is None:
            metric = p.metric

        split_col_char = '_' + split_col + '-'
        project_name_char = prefix + '_FD:'

        stats.loc[i, 'Project_Name'] = p.project_name
        stats.loc[i, 'Project_ID'] = p.id
        stats.loc[i, split_col] = p.project_name.split(split_col_char)[1]
        stats.loc[i, 'FD'] = p.project_name.split(project_name_char)[1].split('_FDW:')[0]
        stats.loc[i, 'FDW'] = p.project_name.split('_FDW:')[1].split(split_col_char)[0]

        m = get_top_models_from_projects([p], n_models=1, metric=metric)[0]
        stats.loc[i, 'Model_Type'] = m.model_type
        stats.loc[i, 'Model_ID'] = m.id

    stats['Series'] = stats[split_col].map(cluster_to_series_map)
    return stats


def get_or_request_predictions(
    models,
    scoring_df,
    ts_settings,
    project_stats=None,
    start_date=None,
    end_date=None,
    forecast_point=None,
    retrain=True,
):

    series_id = ts_settings['series_id']

    models_to_predict_on = []
    retrain_jobs = []
    predict_jobs = []
    project_dataset_map = {}

    for m in models:
        print(m)
        p = dr.Project.get(m.project_id)
        start_date = dr.DatetimePartitioning.get(p.id).available_training_start_date
        print('Start Date: ', start_date)
        print(f'Uploading scoring dataset for Project {p.project_name}')

        # only upload if necessary
        if m.project_id not in project_dataset_map:
            p.unlock_holdout()
            series = project_stats.loc[project_stats['Model_ID'] == m.id, 'Series'][0]
            df = scoring_df.loc[scoring_df[series_id].isin(series), :]
            pred_dataset = p.upload_dataset(df, forecast_point=forecast_point)
            project_dataset_map[m.project_id] = pred_dataset.id

        if retrain:
            try:
                new_model_job = m.request_frozen_datetime_model(
                    training_start_date=start_date, training_end_date=end_date
                )
                retrain_jobs.append(new_model_job)
                print(
                    f'Retraining M{m.model_number} from {start_date} to {end_date} in Project {p.project_name}'
                )
            except dr.errors.JobAlreadyRequested:
                print(
                    f'M{m.model_number} in Project {p.project_name} has already been retrained through the holdout'
                )
                models_to_predict_on.append(m)
        else:
            models_to_predict_on.append(m)

        for job in retrain_jobs:
            models_to_predict_on.append(job.get_result_when_complete(max_wait=10000))

    for model in models_to_predict_on:
        p = dr.Project.get(model.project_id)
        print(f'Getting predictions for M{model.model_number} in Project {p.project_name}\n')
        predict_jobs.append(model.request_predictions(project_dataset_map[model.project_id]))

    preds = [pred_job.get_result_when_complete() for pred_job in predict_jobs]

    predictions = pd.DataFrame()
    for i in range(len(preds)):
        predictions = predictions.append(preds[i])

    print('Finished computing and downloading predictions')
    return predictions


def merge_preds_and_actuals(preds, actuals, ts_settings):
    series_id = ts_settings['series_id']
    date_col = ts_settings['date_col']

    preds['timestamp'] = pd.to_datetime(preds['timestamp']).dt.tz_localize(None)
    preds_and_actuals = preds.merge(
        actuals, how='left', left_on=['series_id', 'timestamp'], right_on=[series_id, date_col]
    )
    return preds_and_actuals
