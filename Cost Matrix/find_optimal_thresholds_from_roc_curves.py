"""
Helper library to work with DataRobot Python3 API client
to cycle through all models of a project and identify the
optimal threshold given an objective function

find_optimal_thresholds works at the project level
find_optimal_threshold works on a single model
create_fbeta_function is an example objective function using fscore
create_payoff_function is an example objective based on values on the confusion matrix
"""


def find_optimal_threshold(model, source, objective_function, objective_label='objective', maximize=True):
    """Returns the dict of roc_point data point for a single model that optimizes
    the objective function.
       
    Args:
        model (datarobot.Model): datarobot project object to handle remaining class calls
        source (str): eligible values in ('validation', 'crossValidation', 'holdout')
        objective_function (function): must return a single float value
        objective_label (str): key label of the objective function output.  Defaults to "objective"
        minimize (boolean): set to True if objective is to be minimized.  Defaults to True.
    Returns:
        dict of roc points from the model in project with additional fields:
            (project_id, model_id, <objective_value>)
    """
    output_values = {'project_id': model.project_id,
                     'model_id': model.id,
                     objective_label: None}
    for roc_curve in model.get_all_roc_curves():
        if roc_curve.source == source:
            break
    else: # source not found
        return output_values
    
    best_index = 0
    best_value = objective_function(roc_curve.roc_points[0])
    
    for current_index, roc_point in enumerate(roc_curve.roc_points[1:], 1):
        current_value = objective_function(roc_point)
        if maximize:
            if current_value > best_value:
                best_value = current_value
                best_index = current_index
        else:
            if current_value < best_value:
                best_value = current_value
                best_index = current_index
    output_values.update(roc_curve.roc_points[best_index])
    output_values[objective_label] = best_value
    return output_values


def find_optimal_thresholds(project, source, objective_function, objective_label='objective', maximize=True):
    """Returns the roc_point data point for each model in the project that optimizes
    the objective function.
       
    Args:
        project (datarobot.Project): datarobot project object to handle remaining class calls
        source (str): eligible values in ('validation', 'crossValidation', 'holdout')
        objective_function (function): must return a single float value
        objective_label (str): key label of the objective function output.  Defaults to "objective"
        minimize (boolean): set to True if objective is to be minimized.  Defaults to True.
    Returns:
        List of roc points from each model in project with additional fields:
            (project_id, model_id, <objective_value>)
    """
    output_values = []
    for model in project.get_models():
        record = find_optimal_threshold(model, source, objective_function, objective_label, maximize)
        output_values.append(record)
    return output_values


def create_fbeta_function(beta):
    """Returns the f score function with designated beta to be applied to
    the roc points provided by the datarobot python api RocCurve roc_points
    single row.
       
    Args:
        beta (float): values > 0.  1 if you want the f1score function
        recalls (numpy.ndarray): An array of recall scores
        beta (float): beta used in fscore.  For f1 score, beta=1
    Returns:
        function: the function that can calculate the fscore from a row of the
            roc_points, that looks like example below:
            {'accuracy': 0.60375,
             'f1_score': 0.0031446540880503146,
             'false_negative_score': 634,
             'false_positive_rate': 0.0,
             'false_positive_score': 0,
             'fraction_predicted_as_negative': 0.999375,
             'fraction_predicted_as_positive': 0.000625,
             'lift_negative': 1.0006253908692933,
             'lift_positive': 2.5196850393700787,
             'matthews_correlation_coefficient': 0.0308285119300898,
             'negative_predictive_value': 0.6035021888680425,
             'positive_predictive_value': 1.0,
             'threshold': 0.8830509781837463,
             'true_negative_rate': 1.0,
             'true_negative_score': 965,
             'true_positive_rate': 0.0015748031496062992,
             'true_positive_score': 1}
    """
    def fbeta_function(roc_point):
        precision = roc_point['positive_predictive_value']
        recall = roc_point['true_positive_rate']
        try:
            score = (1+beta**2) * (precision*recall) / ((beta**2 * precision) + recall)
        except ZeroDivisionError:
            score = 0
        return score
    return fbeta_function


def create_payoff_function(tn, fn, tp, fp):
    """Returns the payoff function that given roc points will calculate the net payoff
    
    Args:
        tn (float): net value of a true negative
        fn (float): net value of a false negative
        tp (float): net value of a true positive
        fp (float): net value of a false positive
        
    Returns:
        float of net value of values multiplied by the confusion matrix values
    """
    def payoff_function(roc_point):
        tns = roc_point['true_negative_score']
        fns = roc_point['false_negative_score']
        tps = roc_point['true_positive_score']
        fps = roc_point['false_positive_score']
        return tn*tns + fn*fns + tp*tps + fp*fps
    
    return payoff_function
