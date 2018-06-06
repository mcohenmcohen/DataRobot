import pandas as pd
import datarobot


def get_results(proj):
    '''
    Generates a summary of all model performances and put it into a DataFrame

    Args: A DataRobot project object

       returns: A dataframe sorted by log loss for cross-validation
    '''
    project = dr.Project.get(project_id=proj)
    # extract featurelist
    feature_lists = project.get_featurelists()

    # get informative features, the default for autopilot
    # you could update this to your feature list
    f_list = [lst for lst in feature_lists if lst.name == 'Informative Features'][0]

    # get models
    models = project.get_models()
    flist_models = [model for model in models if model.featurelist_id == f_list.id]

    # print results
    val_scores = pd.DataFrame([{'model_type': model.model_type,
                                'blueprint info': model.blueprint,
                                'model_id': model.id,
                                'sample_pct': model.sample_pct,
                                'featurelist': model.featurelist_name,
                                'val_logloss': model.metrics['LogLoss']['validation'],
                                'cross_val_logloss': model.metrics['LogLoss']['crossValidation']}
                                 for model in flist_models if model.metrics['LogLoss'] is not None])

    return val_scores.sort_values(by='cross_val_logloss')
modelframe = get_results(project.id)
modelframe
