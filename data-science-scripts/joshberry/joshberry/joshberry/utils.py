def reduce_mem_usage(df, aggressive=False, verbose=True):
    '''
    Returns a dataframe which is the same as the input dataframe, except taking up a smaller
    amount of memory.

            Parameters:
                    df (DataFrame): a pandas dataframe
            Returns:
                    df (DataFrame): a pandas dataframe
    '''
    import numpy as np
    import pandas as pd
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if aggressive == True:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} MB ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def combine_to_list(inlist, joiner):
    '''
    Combines a list of items and returns a list with a custom separator

            Parameters:
                    inlist (list): a list of items
                    joiner (str) : a custom separator for the list
            Returns:
                    list
    '''
    l = [
        joiner.join(ts)
        for ts in [t for t in list(itertools.combinations(inlist, 2)) if t[0] != t[1]]
    ]

    return l



def chunks(l, n, randomize=True):
    '''
    Chunks a long list into pieces of size n

            Parameters:
                    l (list): a list
                    n (int) : size n for max size of each chunk
            Returns:
                    list of chunks (list of lists)
    '''
    # consider randomizing the list first
    if randomize == True:
        import random
        random.shuffle(l)
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]




def xldate_as_datetime(xldate, datemode):
    """
    ##
    # Convert an Excel number (presumed to represent a date, a datetime or a time) into
    # a Python datetime.datetime
    # @param xldate The Excel number
    # @param datemode 0: 1900-based, 1: 1904-based.
    # <br>WARNING: when using this function to
    # interpret the contents of a workbook, you should pass in the Book.datemode
    # attribute of that workbook. Whether
    # the workbook has ever been anywhere near a Macintosh is irrelevant.
    # @return a datetime.datetime object, to the nearest_second.
    # <br>Special case: if 0.0 <= xldate < 1.0, it is assumed to represent a time;
    # a datetime.time object will be returned.
    # <br>Note: 1904-01-01 is not regarded as a valid date in the datemode 1 system; its "serial number"
    # is zero.
    # @throws XLDateNegative xldate < 0.00
    # @throws XLDateAmbiguous The 1900 leap-year problem (datemode == 0 and 1.0 <= xldate < 61.0)
    # @throws XLDateTooLarge Gregorian year 10000 or later
    # @throws XLDateBadDatemode datemode arg is neither 0 nor 1
    # @throws XLDateError Covers the 4 specific errors
    """
    if datemode not in (0, 1):
        raise XLDateBadDatemode(datemode)
    if xldate == 0.00:
        return datetime.time(0, 0, 0)
    if xldate < 0.00:
        raise XLDateNegative(xldate)
    xldays = int(xldate)
    frac = xldate - xldays
    seconds = int(round(frac * 86400.0))
    assert 0 <= seconds <= 86400
    if seconds == 86400:
        seconds = 0
        xldays += 1
    if xldays >= _XLDAYS_TOO_LARGE[datemode]:
        raise XLDateTooLarge(xldate)

    if xldays == 0:
        # second = seconds % 60; minutes = seconds // 60
        minutes, second = divmod(seconds, 60)
        # minute = minutes % 60; hour    = minutes // 60
        hour, minute = divmod(minutes, 60)
        return datetime.time(hour, minute, second)

    if xldays < 61 and datemode == 0:
        raise XLDateAmbiguous(xldate)

    return (
        datetime.datetime.fromordinal(xldays + 693594 + 1462 * datemode)
        + datetime.timedelta(seconds=seconds)
        )
    
    
def xldatestring_as_datetime(df, datecols):
    '''
    Fixes the y/m/dd excel string date and makes it a real date

            Parameters:
                    df (DataFrame)  : original dataframe
                    datecols (list) : a list of date column names (str)
            Returns:
                    DataFrame
    '''
    import pandas as pd
    
    df = df.copy(deep=True)
    
    for c in datecols:
        msg = "-" * 10
        msg = msg + "\nORIGINAL: " + str(df[[c]].iloc[0].values.tolist()[0])
        df[c] = pd.to_datetime(df[c], format='%m/%d/%y').dt.strftime('%Y-%m-%d %H:%M:%S')
        msg = msg + "\nMODIFIED: " + str(df[[c]].iloc[0].values.tolist()[0])
        print(msg)
    return df

def check_topn(df, topn, id_col_list, prob_col_list, actual_col_str, actuals_mapping_dict):
    '''
    Takes data with rows as rows, multiclass probabilities as columns, and an actual.
    Returns a dataframe with a flag for whether actual was within Top N probabilities

            Parameters:
                    df  (DataFrame)                : main dataframe
                    topn (int)                     : integer N of Top N probabilities
                    id_col_list (list)             : list of id columns (for joining)
                    prob_col_list (list)           : list of probability columns
                    actual_col_str (str)           : column name of the target (actual)
                    actuals_mapping_dict (dict)    : dictionary mapping actual value to the name of the correponding probability column
            Returns:
                    DataFrame
    '''
    r = df[prob_col_list].rank(axis=1, ascending=False).astype(int)
    r[actual_col_str] = df[actual_col_str]
    r[id_col_list] = df[id_col_list]
    r['is_in_top_n'] = r.apply(lambda row: row[actuals_mapping_dict[row[actual_col_str]]] <= topn, axis=1).astype(int)
    
    return_cols = id_col_list + ['is_in_top_n']
    
    res = df.merge(r[return_cols], how='inner', on=id_col_list)
    
    return res

#### Audio Notifications for long running cells in jupyter lab
    '''
    By executing this code, we are enabling some audio alerts to play whenever a long running cell
    completes running. Anything less than the threshold will not play the alert.
    The threshold is configured below
    '''
PLAY_AUDIO_THRESHOLD = 60  # seconds
from time import time
from IPython import get_ipython
from IPython.display import Audio, display

class InvisibleAudio(Audio):
    def _repr_html_(self):
        audio = super()._repr_html_()
        audio = audio.replace('<audio', f'<audio onended="this.parentNode.removeChild(this)"')
        return f'<div style="display:none">{audio}</div>'

class Beeper:

    def __init__(self, threshold, **audio_kwargs):
        self.threshold = threshold
        self.start_time = None    # time in sec, or None
        self.audio = audio_kwargs

    def pre_execute(self):
        if not self.start_time:
            self.start_time = time()

    def post_execute(self):
        end_time = time()
        if self.start_time and end_time - self.start_time > self.threshold:
            audio = InvisibleAudio(**self.audio, autoplay=True)
            display(audio)
        self.start_time = None

# camera sound: https://www.soundjay.com/mechanical/sounds/camera-shutter-click-04.mp3
# typewriter: https://www.soundjay.com/communication/sounds/typewriter-backspace-1.mp3
# bell: https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3
beeper = Beeper(PLAY_AUDIO_THRESHOLD, url='https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3')

ipython = get_ipython()
ipython.events.register('pre_execute', beeper.pre_execute)
ipython.events.register('post_execute', beeper.post_execute)


def chunky(df):
    '''
    Uses a DR model that automatically calculates the best chunk-size for a to_csv operation

            Parameters:
                    df (DataFrame)  : original dataframe
            Returns:
                    int
    '''
    import psutil
    import pandas as pd
    import numpy
    import os


    
    def get_cpu_stats():
        stats={}
        stats['cpu_name'] = os.uname()[1]
        stats['mem_total'] = psutil.virtual_memory()._asdict()['total']
        stats['mem_avail'] = psutil.virtual_memory()._asdict()['available']
        stats['mem_free'] = psutil.virtual_memory()._asdict()['free']
        stats['mem_pctavail'] = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
        stats['cpu_virt_count'] = psutil.cpu_count(logical=True)
        stats['cpu_core_count'] = psutil.cpu_count(logical=False)
        stats['sys_boot_time'] = psutil.boot_time()
        return stats

    def get_df_stats(df):
        stats={}
        stats['df_size'] = df.size
        stats['df_rows'] = df.shape[0]
        stats['df_cols'] = df.shape[1]
        stats['df_mem_used'] = df.memory_usage(deep=True).sum()
        stats['df_row_size'] = df.memory_usage(deep=True).sum() / df.shape[0]
        return stats

    def get_stats(df):
        cpustats = get_cpu_stats()
        dfstats = get_df_stats(df)
        dfstats.update(cpustats)
        return dfstats
    
    def convert(o):
        if isinstance(o, numpy.int64): return int(o)  
        raise TypeError
        
    toscore = get_stats(df)
    API_KEY = os.environ.get("DR_API_TOKEN")
    DATAROBOT_KEY = '544ec55f-61bf-f6ee-0caf-15c7f919a45d'
    
    import sys
    import json
    import requests
    import yaml

    API_URL = 'https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/predictions'
    DEPLOYMENT_ID = '607ddfecae9b700920d61f95'
    
    headers = {'Content-Type': 'application/json; charset=UTF-8','Authorization': 'Bearer {}'.format(API_KEY),'DataRobot-Key': DATAROBOT_KEY}
    #print(headers)
    url = API_URL.format(deployment_id=DEPLOYMENT_ID)
    
    data = json.dumps([toscore], default=convert)
    #print(data)
    # Make API request for predictions
    predictions_response = requests.post(
        url,
        data=data,
        headers=headers
    )    
    #print(predictions_response.text)
    pred = json.loads(predictions_response.text)['data'][0]['prediction']
    print(f'Chunky optimized your write speed by using a chunksize of {int(pred)}.')
    return int(pred)