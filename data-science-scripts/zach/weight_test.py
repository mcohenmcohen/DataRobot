from __future__ import division
import numpy as np

def calculate_mean(maj_count, min_count, maj_weight=None, min_weight=None, zero_inflated=False, seed=42):
    majority_y = np.zeros(maj_count)
    if zero_inflated:
        rs = np.random.RandomState(seed=seed)
        minority_y = rs.gamma(shape=7.5, scale=1, size=min_count)
    else:
        minority_y = np.ones(min_count)
    y = np.hstack([majority_y, minority_y])

    weights = None
    if maj_weight and min_weight:
        maj_weight = np.full(maj_count, maj_weight)
        min_weight = np.full(min_count, min_weight)
        weights = np.hstack([maj_weight, min_weight])

    return np.average(y, weights=weights)

def test_weights(maj_count, min_count, rate):

    down_sampled_min_count = min_count
    down_sampled_maj_count = int(maj_count * rate)

    orginal_row_count = min_count + maj_count
    down_sampled_row_count = down_sampled_min_count + down_sampled_maj_count

    # Old weights
    # Note that DSS rounds to 2 decimals
    old_min_weight = np.round(down_sampled_row_count / orginal_row_count, 2)
    old_maj_weight = np.round(down_sampled_row_count / (rate * orginal_row_count), 2)

    # New weights
    # Note that DSS rounds to 2 decimals
    new_min_weight = np.round(1, 2)
    new_maj_weight = np.round(1 / rate, 2)

    # Check means for classification
    # Note np.testing.assert_almost_equal checks to 7 decimals by default
    orginal_mean = calculate_mean(maj_count, min_count)
    old_mean = calculate_mean(down_sampled_maj_count, down_sampled_min_count, old_maj_weight, old_min_weight)
    new_mean = calculate_mean(down_sampled_maj_count, down_sampled_min_count, new_maj_weight, new_min_weight)

    np.testing.assert_almost_equal(orginal_mean, old_mean, decimal=1)  # Old mean has more rounding error
    np.testing.assert_almost_equal(orginal_mean, new_mean, decimal=4)  # New mean has less rounding error

    # Check means for regression
    orginal_mean = calculate_mean(maj_count, min_count, zero_inflated=True)
    old_mean = calculate_mean(down_sampled_maj_count, down_sampled_min_count, old_maj_weight, old_min_weight, zero_inflated=True)
    new_mean = calculate_mean(down_sampled_maj_count, down_sampled_min_count, new_maj_weight, new_min_weight, zero_inflated=True)

    np.testing.assert_almost_equal(orginal_mean, old_mean, decimal=0)  # Old mean has more rounding error
    np.testing.assert_almost_equal(orginal_mean, new_mean, decimal=4)  # New mean has less rounding error

    # Check minority count for classificaiton (new weights only)
    new_min_count = down_sampled_min_count * new_min_weight
    np.testing.assert_almost_equal(min_count / new_min_count, 1, decimal=3)  # Check the 2 counts are almost the same

    # Check majorty count for classificaiton (new weights only)
    # Divide by 10, to only check the 10's place and up.
    # E.g. we'll call 1266644 vs 1266641 a match, even though the 1s place differs
    new_maj_count = down_sampled_maj_count * new_maj_weight
    np.testing.assert_almost_equal(maj_count / new_maj_count, 1, decimal=3)  # Check the 2 counts are almost the same


for maj_count, min_count, rate in [
    (1266644, 8358, 0.12),  # Cigna case
    (134651334, 1324, 0.00007), # Note this is a huge dataset: 100 million+ rows
    (10000000, 1, 0.001),
    (10, 1, 0.1),
    (100, 1, 0.1),
    (1000, 1, 0.1),
]:
    print((maj_count, min_count, rate))
    test_weights(maj_count, min_count, rate)
