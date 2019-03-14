import datarobot as dr
import pandas as pd
import numpy as np
import time
url = "https://s3.amazonaws.com/datarobot_public_datasets/"
files = [ "DR_Demo_AML_Alert.csv"]
targets = [ "SAR"]

np.random.seed(369)
# Rather than waiting for predictions, I send off all the predict jobs first so they can be run in parallel
# and collect predictions later. You might have to increase the sleep time if the file is big.
def make_predictions(models, dataset_id):
    jobs = []
    for model in models:
        predict_job = model.request_predictions(dataset_id)
        jobs.append(predict_job.id)
    return(jobs)

for i in range(len(files)):
    dataset = pd.read_csv(url + files[i])
    project = dr.Project.create(project_name=files[i], sourcedata=dataset)
    project.unlock_holdout()
    project.set_target(target=targets[i], mode=dr.AUTOPILOT_MODE.MANUAL, worker_count=7)
    counter = 0
    for bp in [bp for bp in project.get_blueprints() if 'Anomaly' in bp.model_type]:
        counter += 1
        project.train(bp, sample_pct=100)
    original_dataset = project.upload_dataset(dataset)
    orig_preds = []
    models = project.get_models()
    while len(models) < counter:
        time.sleep(120)
        models = project.get_models()
    models_string = [str(m).split('(', 1)[1].split(')')[0] for m in models]
    job_ids = make_predictions(models, original_dataset.id)
    for model in range(len(models)):
        time.sleep(120)
        prediction = dr.PredictJob.get_predictions(project.id, predict_job_id=job_ids[model])
        prediction = prediction.iloc[:,0]
        orig_preds.append(prediction)
    orig_preds = pd.DataFrame(orig_preds).transpose()
    orig_preds.columns = models_string
    filename = "preds_" + files[i]
    orig_preds.to_csv(filename, index=False)
