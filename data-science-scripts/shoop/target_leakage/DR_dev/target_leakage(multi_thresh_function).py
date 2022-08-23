# NOTE: This is mainly for reference purposes, to record my dev experimentation progress results.
# If you want to get results from local datarobot app as seen in ace_results.csv,
# then paste this helper function into MM/engine/data_quality/target_leakage.py

# NOTE2: Don't forget to paste the following dev code under get_leakage_type() in eda_multi.py
# leakage_type_thresholds = get_leakage_type_multi_thresholds(ace_score)
# eda_report['target_leakage_metadata_thresholds'] = leakage_type_thresholds


def get_leakage_type_multi_thresholds(ace_normalized_wrt_target):
    """
    Gets leakage type results for different threshold values. Current method: normalized ACE score.
    The amount of thresholds is more than what we have in config.engine.AceLeakageThreshold.

    Parameters
    ----------
    ace_normalized_wrt_target: float
        ACE score that is normalized with respect to target (0.0 to 1.0 range)

    Returns
    -------
    leakage_types_multi_thresholds: dict of float keys and common.enum.TargetLeakageType values
    """

    high_threshold = AceLeakageThreshold.HIGH_RISK  # 0.975
    moderate_threshold = AceLeakageThreshold.MODERATE_RISK  # 0.85

    def _frange(start, stop, step):
        i = start
        while i > stop:
            yield round(i, 3)
            i -= step

    experimental_moderate_thresholds = [i for i in _frange(1.0, 0.0, 0.025)]
    threshold_pairs = [(thresh, high_threshold) for thresh in experimental_moderate_thresholds]
    leakage_types = [None for _ in range(len(threshold_pairs))]

    # leakage_threshold is a tuple, where:
    # - leakage_threshold[0] is the MODERATE_RISK threshold,
    # - leakage_threshold[1] is the HIGH_RISK threshold (for now, this is always 0.975)
    for idx, leakage_threshold in enumerate(threshold_pairs):
        # if there is at least one HIGH_RISK detected,
        # then all leakage_types for all thresholds should be HIGH_RISK
        if TargetLeakageType.HIGH_RISK in leakage_types:
            leakage_types[idx] = TargetLeakageType.HIGH_RISK
            continue
        if (
            ace_normalized_wrt_target >= leakage_threshold[1]
            and leakage_threshold[1] >= high_threshold
        ):
            leakage_types[idx] = TargetLeakageType.HIGH_RISK
            continue
        if ace_normalized_wrt_target >= leakage_threshold[0]:
            leakage_types[idx] = TargetLeakageType.MODERATE_RISK
            continue
        else:
            leakage_types[idx] = TargetLeakageType.FALSE

    ace_leakage_thresholds_results = [
        {'threshold_pair': threshold_pair, 'leakage_type': leakage_type}
        for threshold_pair, leakage_type in zip(threshold_pairs, leakage_types)
    ]
    return {'ace_leakage_thresholds': ace_leakage_thresholds_results}
