# paste this into eda_multi.py

def multiclass_ace(
    ace_data, target_name, categoricals, weight, eda_report, target_metric, **kwargs
):
    """ Calculate multiclass ACE score as weighted average for each target class present in
    EDA sample. Averaging weights are "class count" for unweighted projects and
    "sum of class weights" for weighted.

    Parameters
    ----------
    ace_data : pandas.DataFrame
        Dataframe containing all the information for ACE calculation.
        Usually - column we calculate ACE for, target column and weights(if present).
    target_name : str
        Name of the target column.
    categoricals : list of str
        List of the columns that ar categoricals.
    weight : str
        Name of the weights column.
    kwargs : dict
        Other optional parameters that will be passed to tesla ACE calls without change

    Returns
    -------
    list of list of float
        Ace scores calculated as weighted average for binary scores for each present target class.
        Format of the results can be seen as described below. As of 2020-01-21 we always call ACE
        for each column separately, so returns single element list. If we ever change this -
        code below will need to be updated.
        (copy from tesla.ace)
        List where each item is a list containing the correlations between each column
        of `x` and `target` and the correlation of a mean-only prediction and the target
    """
    original_target = ace_data[target_name]
    result = [[0, 0]]
    target_values = original_target.unique()
    target_values_raw_ace = {}
    target_values_normalized_ace = {}
    target_values_gininorm_ace = {}
    target_values_acc_ace = {}

    with pandas.option_context('mode.chained_assignment', None):
        for target_value in target_values:  # for eg. virginica, setosa, versicolor (0.0, 1.0, 2.0)
            bin_target = original_target == target_value
            ace_data[target_name] = bin_target
            bin_ace = ace(ace_data, target_name, categoricals, weight=weight, **kwargs)
            # gini norm ACE score of the bin_ace, bin_target
            gini_norm_dir = metrics.direction_by_name("Gini Norm")
            accuracy_dir = metrics.direction_by_name("Accuracy")
            gini_norm_bin_ace = ace(
                ace_data,
                target_name,
                categoricals,
                cv=True,
                metric=gini_norm,
                metric_dir=gini_norm_dir,
                K=-1,
                weight=weight,
            )
            accuracy_bin_ace = ace(
                ace_data,
                target_name,
                categoricals,
                cv=True,
                metric=accuracy,
                metric_dir=accuracy_dir,
                K=-1,
                weight=weight,
            )
            target_values_raw_ace[str(target_value)] = bin_ace[0][0]
            target_values_normalized_ace[str(target_value)] = metrics.normalize_ace_score(
                target_metric, bin_ace[0][0], bin_ace[0][1]
            )
            target_values_gininorm_ace[str(target_value)] = gini_norm_bin_ace[0][0]
            target_values_acc_ace[str(target_value)] = accuracy_bin_ace[0][0]

            if weight is not None:
                k = np.average(bin_target, weights=ace_data[weight])
            else:
                k = np.mean(bin_target)
            result[0][0] += k * bin_ace[0][0]
            result[0][1] += k * bin_ace[0][1]
        ace_data[target_name] = original_target
        eda_report['multiclass_raw_ace_vals'] = target_values_raw_ace
        eda_report['multiclass_normlized_ace_vals'] = target_values_normalized_ace
        eda_report['multiclass_gininorm_ace_vals'] = target_values_gininorm_ace
        eda_report['multiclass_acc_ace_vals'] = target_values_gininorm_ace

    return result

# Don't forget to edit the ace_method() part too
#
# info = ace_method(
#     ace_data,
#     target_name,
#     categoricals,
#     cv=True,
#     metric=mfunc,
#     metric_dir=mdir,
#     K=-1,
#     weight=adj_weight_name,
#     eda_report=eda_report,
#     target_metric=target_metric,
# )
