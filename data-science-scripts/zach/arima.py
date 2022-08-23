import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from itertools import product, izip
from joblib import Parallel, delayed
from multiprocessing import Pool, freeze_support
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
import warnings
import random


#%matplotlib inline
%matplotlib

#http://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_stata.html
#from dismalpy import SARIMAX

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


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('/Users/zachary/workspace/data-science-scripts/zach/AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'])
data = data.set_index('Month')

ts = data['#Passengers']

# ts_train = ts[0:132]
# ts_test = ts[132:144]

ts_train = ts['1949-01-01':'1958-12-01']
ts_test = ts['1959-01-01':'1960-12-01']

#Define models
#http://stackoverflow.com/a/5228294
MIN_ORDER = 0
MAX_ORDER = 10
def my_product(dicts):
    return (dict(izip(dicts, x)) for x in product(*dicts.itervalues()))
model_params = dict(
    p=range(5),
    d=range(2),
    q=range(5),
    P=range(2),
    D=range(1),
    Q=range(2),
    period = [1, 4, 12],
    trend=['n', 'c', 't', 'ct']
)

#[1,1,1] squared
#[1,1,1,1] cubed

job_list = my_product(model_params)
job_list = [x for x in job_list]
job_list = [{
    'main_model': (x['p'], x['d'], x['q']),
    'seasonal_model': (x['P'], x['D'], x['Q'], x['period']),
    'order': x['p'] + x['q'] + x['P'] + x['Q'],
    'seasonal_order': x['P'] + x['Q'],
    'period': x['period'],
    'trend': x['trend']
    } for x in job_list]
job_list = [x for x in job_list if (x['order']) > MIN_ORDER]
job_list = [x for x in job_list if (x['order']) < MAX_ORDER]
job_list = [x for x in job_list if (x['seasonal_order']) > 0 or x['period'] > 1]
job_list[0]

#Shuffle job list
random.seed(42)
random.shuffle(job_list)

#Define and test the fit method
def fit_model(x, return_model=False, ic = 'bic'):
    model = SARIMAX(ts_train, order=x['main_model'], seasonal_order=x['seasonal_model'], trend=x['trend'], enforce_invertibility=False, enforce_stationarity=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge. Check mle_retvals")
        fit = model.fit(disp=False)
    if return_model:
        return fit
    return getattr(fit, ic)
fit_model(job_list[0])

#Fit models in parallel
bics = parallel_process(job_list, fit_model, n_jobs=8)

#Select best based on bic
best = np.argmin(bics)
print(job_list[best])
print(bics[best])
final_model = fit_model(job_list[best], return_model=True)
final_model.summary()
# pred = final_model.predict(start=132, end=143)

pred_test = final_model.predict(start='1958-12-01', end='1960-12-01')
print(np.sqrt(np.mean((pred_test - ts_test) ** 2)))

pred_train = final_model.predict()
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(ts)
ax1.plot(pred_train,label=1)
ax1.plot(pred_test,label=2)
