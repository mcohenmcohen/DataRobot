import warnings
import random
import pandas as pd
import numpy as np
from itertools import product, izip
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.stattools import adfuller

#http://danshiebler.com/2016-09-14-parallel-progress-bar/
def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

def dict_product(dicts):
    return (dict(izip(dicts, x)) for x in product(*dicts.itervalues()))

#Define fit method
def fit_SARIMAX(x, return_model=False):
    model = SARIMAX(
        x['ts_train'], order=x['main_model'], seasonal_order=x['seasonal_model'],
        trend=x['trend'], enforce_invertibility=False, enforce_stationarity=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge. Check mle_retvals")
        fit = model.fit(disp=False)
    if return_model:
        return fit
    return getattr(fit, x['ic'])

def auto_arima(
    time_series, # The input
    max_p=4, max_d=1, max_q=4, # Specification of the main (Ar, I, Ma) model
    max_P=1, max_D=1, max_Q=1, # Specification of the seasona; (Ar, I, Ma) model
    MIN_ORDER = 1, # Can't fit a model with order of 0
    MAX_ORDER = 10, # Cap the max p + q + P + Q
    try_period = [1, 4, 12], # Automatically detect the period!
    try_trend = ['nc', 'c', 'ct', 'ctt'], # n=regular, c=mean, ct=mean+drift, ctt=mean+drift+drift^2
    ic = 'bic', # aic or bic
    n_jobs=8, # Parallel jobs
    ):

    #Dictionary of parameters to try
    model_params = dict(
        p=range(max_p + 1),
        d=range(max_d + 1),
        q=range(max_q + 1),
        P=range(max_P + 1),
        D=range(max_D + 1),
        Q=range(max_Q + 1),
        period = try_period,
        trend = try_trend,
        ic = [ic]
    )

    #Unit root test
    no_unit_root_pvalue = {}
    for t in try_trend:
        no_unit_root_pvalue[t] = adfuller(ts_train, regression=t, autolag=ic)[1]

    #Choose trend
    model_params['trend'] = min(no_unit_root_pvalue, key=no_unit_root_pvalue.get)
    best_p = no_unit_root_pvalue[model_params['trend']]

    #Rename trend to work with the SARIMAX function
    if model_params['trend'] == 'nc':
        model_params['trend'] = 'n'
    if model_params['trend'] == 'ctt':
        model_params['trend'] = [[1,1,1]]

    #Choose integration
    if best_p > .50: # Pretty sure no unit root
        model_params['d'] = [0]
    elif best_p < .05: # Pretty sure unit root
        model_params['d'] = [1]

    #Product of all the different parameters
    job_list = dict_product(model_params)
    job_list = [x for x in job_list]
    job_list = [{
        'ts_train': time_series,
        'main_model': (x['p'], x['d'], x['q']),
        'seasonal_model': (x['P'], x['D'], x['Q'], x['period']),
        'order': x['p'] + x['q'] + x['P'] + x['Q'],
        'seasonal_order': x['P'] + x['Q'],
        'period': x['period'],
        'trend': x['trend'],
        'ic': ic
        } for x in job_list]
    job_list = [x for x in job_list if (x['order']) >= MIN_ORDER]
    job_list = [x for x in job_list if (x['order']) <= MAX_ORDER]

    #Skip seasonal models when seasonality is 1
    job_list = [x for x in job_list if (x['seasonal_order']) > 0 or x['period'] > 1]

    #Shuffle job list
    #Simpler models run faster, so this gives us more accurate timings
    random.seed(42)
    random.shuffle(job_list)

    #Fit models in parallel
    bics = parallel_process(job_list, fit_SARIMAX, n_jobs=8)

    #Select best based on bic
    best = np.argmin(bics)

    #Refit final model
    #We should be able to skip this step
    #but for some reason fitting the models in parallel messes up their ability to predict
    final_model = fit_SARIMAX(job_list[best], return_model=True)

    return final_model

if __name__ == "__main__":
    import matplotlib.pylab as plt
    from datetime import datetime

    # base_dir = '/Users/zachary/workspace/data-science-scripts/zach/'
    # n_jobs = 8
    base_dir = '/home/zach/'
    n_jobs = 72

    #Air passengers
    data = pd.read_csv(base_dir + 'AirPassengers.csv')
    data['Month'] = pd.to_datetime(data['Month'])
    data = data.set_index('Month')
    ts = data['#Passengers']
    ts_train = ts['1949-01-01':'1958-12-01']
    ts_test = ts['1959-01-01':'1960-12-01']

    final_model = auto_arima(ts_train, MAX_ORDER=4, max_d=1, n_jobs=n_jobs)
    print final_model.summary()

    pred_test = final_model.predict(start='1958-12-01', end='1960-12-01')
    print(np.sqrt(np.mean((pred_test - ts_test) ** 2)))

    %matplotlib
    pred_train = final_model.predict()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(ts)
    ax1.plot(pred_train,label=1)
    ax1.plot(pred_test,label=2)

    #CO2
    ts_train = pd.read_csv(base_dir + 'co2_train.csv')
    ts_test = pd.read_csv(base_dir + 'co2_test.csv')
    ts_train['date'] = pd.to_datetime(ts_train['date'])
    ts_test['date'] = pd.to_datetime(ts_test['date'])
    ts_train = ts_train.set_index('date')['co2']
    ts_test = ts_test.set_index('date')['co2']
    final_model = auto_arima(ts_train, MAX_ORDER=5, max_p=2, max_d=1, max_q=2, try_period = [12], n_jobs=n_jobs)
    print final_model.summary()
    pred_test = final_model.predict(start=ts_train.shape[0], end=ts_train.shape[0]+ts_test.shape[0])
    print(np.sqrt(np.mean((pred_test - ts_test) ** 2)))
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(ts_train)
    ax1.plot(ts_test)
    ax1.plot(pred_test,label=2)
