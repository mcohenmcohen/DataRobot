def calculate_woe_iv(dataset, feature, target):
    '''
    Calculates the Information Value of a specific variable against a binary target.
    Note that this only works for categorical values because we have not bothered to 
    incorporate any binning strategies for numerics

            Parameters:
                    dataset (DataFrame) : DataFrame
                    feature (str)       : name of the feature to analyze
                    target (str)        : name of the target feature
            Returns:
                    Information Value (float)
    '''
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })
        
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    
    #dset = dset.sort_values(by='WoE')
    
    return iv


def add_partition(df,
                  k=5,
                  holdout_size=0.20, 
                  target_variable_name='target', 
                  holdout_indicator_name='holdout_ind'):
    '''
    Modifies a dataset to include a partition indicator using KFold stratified sampling. For this reason
    it is meant for a binary target only.

            Parameters:
                    df (DataFrame)               : DataFrame
                    n (int)                      : number of training partitions (excluding holdout)
                    holdout_size (float)         : percentage of holdout
                    target_variable_name (str)   : name of target variable (binary only)
                    holdout_indicator_name (str) : name of the holdout indicator to be created
            Returns:
                    DataFrame with holdout indicator added
    '''
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold
    
    df = df.copy(deep=True)
    
    with pd.option_context('mode.chained_assignment', None):
        temptrain, holdoutdata = train_test_split(
            df,
            test_size=holdout_size,
            random_state=1337,
            stratify=df[target_variable_name]
        )

        skf = StratifiedKFold(n_splits=k)
        i = 0
        for _ , test_index in skf.split(temptrain,temptrain[target_variable_name]):
            i+=1
            temp = temptrain.iloc[test_index]
            temp[holdout_indicator_name]='T' + str(i)

            if i==1:
                traindata = temp.copy()
            else:
                traindata = traindata.append(temp)

        holdoutdata[holdout_indicator_name]='H'
        result = pd.concat([traindata,holdoutdata])
        
        return result
    
    
    
def add_business_day_cols(df, datecols):
    '''
    Modifies a dataset to include new columns that describe the date in terms of business days remaining and
    elapsed in the month

            Parameters:
                    df (DataFrame)               : DataFrame
                    datecols (list of str)       : number of datecolumns for which to perform the operation

            Returns:
                    DataFrame with new columns added
    '''
    import numpy as np
    import pandas as pd
    
    df = df.copy(deep=True)
    
    #TODO: Lots of error checking regarding date formats could be helpful here
    
    # this function handles when dates are sometimes missing
    def business_days(start, end, weekmask=None):
        mask = pd.notnull(start) & pd.notnull(end)
        start = start.values.astype('datetime64[D]')[mask]
        end = end.values.astype('datetime64[D]')[mask]
        result = np.empty(len(mask), dtype=float)
        if not weekmask is None:
            result[mask] = np.busday_count(start, end, weekmask=weekmask)
        else:
            result[mask] = np.busday_count(start, end)
        result[~mask] = np.nan
        return result
    
    
    for dt in datecols:
        # helper columns (I like to use __ as prefix and then drop all with this prefix at the end)
        df['__begin_of_curr_mo'] = df[dt].values.astype('datetime64[M]')
        df['__begin_of_next_mo'] = (df[dt] + pd.DateOffset(months=1)).values.astype('datetime64[M]')
        # note the handy way of counting days of week and bizdays is the same function with or without a weekday mask
        df['num_sats_exist_in_mo_' + str(dt)] = business_days(df['__begin_of_curr_mo'].astype('datetime64[D]'), df['__begin_of_next_mo'].astype('datetime64[D]'), weekmask='Sat')
        df['num_sats_elapsed_in_mo_' + str(dt)] = business_days(df['__begin_of_curr_mo'].astype('datetime64[D]'), df[dt].astype('datetime64[D]'), weekmask='Sat')
        df['num_sats_remain_in_mo_' + str(dt)] = business_days(df[dt].astype('datetime64[D]'), df['__begin_of_next_mo'].astype('datetime64[D]'), weekmask='Sat')
        df['num_bizdays_exist_in_mo_' + str(dt)] = business_days(df['__begin_of_curr_mo'].astype('datetime64[D]'), df['__begin_of_next_mo'].astype('datetime64[D]'))
        df['num_bizdays_elapsed_in_mo_' + str(dt)] = business_days(df['__begin_of_curr_mo'].astype('datetime64[D]'), df[dt].astype('datetime64[D]'))
        df['num_bizdays_remain_in_mo_' + str(dt)] = business_days(df[dt].astype('datetime64[D]'), df['__begin_of_next_mo'].astype('datetime64[D]'))
    # drop helper columns
    df = df[df.columns.drop(list(df.filter(regex='__')))]
    
    return df

def add_calendar_cols(df, dfEvents, date_col_list, eventname_col='Name', eventdate_col='Date'):
    '''
    References a calendar file and adds variables to the main dataset of how far away each "event" is

            Parameters:
                    dfMain  (DataFrame)  : main dataset with 1 or more date columns to reference
                    dfEvents (DataFrame) : calendar dataset with 2 columns, 1 for date and 1 for name of the event
                    date_col_list (str)  : list of reference dates from the main dataset
                    eventname_col (str)  : name of the column for DATE in the calendar dataset
                    eventdate_col (str)  : name of the column for EVENT NAME in calendar dataset
            Returns:
                    DataFrame
    '''
    import pandas as pd
    import numpy as np
    
    
    # convert date cols to datetime
    for c in date_col_list:
        df[c] = pd.to_datetime(df[c])
    dfEvents[eventdate_col] = pd.to_datetime(dfEvents[eventdate_col])
    
    # function that calculates days until next event
    def calc_days(ddf, dfCal, direction, mainjoinkey, eventjoinkey):

        # mask missing values with the minimum (this is temporary to avoid an error)
        ddf['__temp'] = ddf[mainjoinkey].mask(ddf[mainjoinkey].isnull(), ddf[mainjoinkey].min())
        
        # will need to be sorted
        ddf.sort_values(by=['__temp'], inplace=True, ignore_index=True)

        # calculate days until or since, based on direction (which is passed in)
        s = pd.merge_asof(ddf, dfCal.sort_values(eventjoinkey), left_on=['__temp'], right_on=eventjoinkey, direction=direction)
        s = ((s[eventjoinkey] - s['__temp']).dt.days.abs() * np.where(ddf[mainjoinkey].notnull(), 1, np.NaN))

        return s
    
    # unique list of events
    unique_events = dfEvents[eventname_col].unique().tolist()
    
    # loop in case there are multiple date columns
    for dtcol in date_col_list:
        
        # dataframe of unique dates
        dfDates = pd.DataFrame(df[dtcol].unique(),columns=[dtcol])
        
        # calc days until the next event
        dfDates['days_until_next_event'] = calc_days(dfDates, dfEvents, 'forward', dtcol, eventdate_col)
        dfDates['days_since_last_event'] = calc_days(dfDates, dfEvents, 'backward', dtcol, eventdate_col)
        
        # do the same for each specific event
        for e in unique_events:  
            dfDates[dtcol + '_days_until_' + e] = calc_days(dfDates, dfEvents[dfEvents[eventname_col].eq(e)], 'forward', dtcol, eventdate_col)
            dfDates[dtcol + '_days_since_' + e] = calc_days(dfDates, dfEvents[dfEvents[eventname_col].eq(e)], 'backward', dtcol, eventdate_col)
        
        # merge everything back to the original dataframe    
        result = df.merge(dfDates, how='left', left_on=dtcol, right_on=dtcol)
        
        # drop the temp column
        result.drop('__temp', axis=1, inplace=True)
        
    return result

def sax(df, dtcol, groupbycols, targetcol, n_bins=5, strategy='quantile'):
    '''
    Takes time series data, groups sequential numbers, and represents them in SAX form. 

            Parameters:
                    df  (DataFrame)    : main time-series ready dataset
                    dtcol (str)        : name of column with date
                    groupbycols (list) : list of group by columns to represent multiseries
                    targetcol (str)    : name of the column that is the target
                    n_bins (int)       : number of bins for sax to recognize. must be 2 <= n <= 26
                    strategy (str)     : 'quantile', 'normal' - depending on distrubution of target
            Returns:
                    DataFrame
    '''
    from pyts.approximation import SymbolicAggregateApproximation
    import warnings
    import pandas as pd
    import numpy as np
    
    ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy=strategy)
    
    df = df.sort_values(groupbycols + [dtcol])
    grouped = df.groupby(groupbycols)
    dfsax = grouped[groupbycols + [targetcol]].aggregate(lambda x: tuple(x))
    
    dfsax['sequence_length'] = dfsax[targetcol].apply(lambda x: len(x))
    dfsax['unique_counts_in_sequence'] = dfsax[targetcol].apply(lambda x: len(list(dict.fromkeys(x))))
    unique_lens = dfsax.sequence_length.unique()
    
    enhanced = pd.DataFrame()

    for l in unique_lens:
        if l > n_bins:
            filtered = dfsax[(dfsax['sequence_length']==l) & (dfsax['unique_counts_in_sequence']>1)].copy()
            if len(filtered) > 0:
                with warnings.catch_warnings(record=True):
                    filtered['__temp'] = filtered[targetcol].apply(lambda x: sax.fit_transform(np.array(x).reshape(1,-1)))
                enhanced = enhanced.append(filtered)

    enhanced['sax_txt'] = enhanced['__temp'].apply(lambda x: x.tostring().decode("utf-8"))

    keep_cols = ['sax_txt','sequence_length','unique_counts_in_sequence']
    enhanced = enhanced[keep_cols].reset_index()
    
    for n in range(0,n_bins):
        char = ascii_lowercase[n:n+1]
        # contains a bootleg fix to numpy using x4 bits to store text, which affects length function
        enhanced['pct_bin_' + char] = enhanced['sax_txt'].apply(lambda s: str(s).count(char) / (len(s) / 4))
    
    result = pd.merge(df, enhanced, how='left', on=groupbycols, suffixes=('', '_y'))
    
    drop_cols = [targetcol + '_y']
    
    try:
        result = result.drop(columns=drop_cols)
    except:
        pass

    
    return result


def lag_value(df, dtcol, multiseries_id, targetcol, windowlist):
    '''
    Takes a dataset with multiseries id, date, and a value, and returns a series of lags
    Mostly used on the target
    note: pass unindexed dataset

            Parameters:
                    df  (DataFrame)       : main time-series ready dataset
                    dtcol (str)           : name of column with date
                    multiseries_id (str)  : name of column with multiseries_id
                    targetcol (str)       : name of the column that is the target
                    windowlist (list)     : list of periods for each window to be performed

            Returns:
                    DataFrame
    '''
    import pandas as pd
    import gc
    from tqdm.notebook import tqdm
    import numpy as np
    
    container = []
    df = df.set_index(dtcol).sort_index()
    
    for seriesId, seriesDf in tqdm(df.groupby(multiseries_id)):
        seriesDfCopy = seriesDf.sort_index()
        nPeriods = seriesDfCopy.shape[0]
        
        for w in windowlist:
            NonzeroMeanLog = np.zeros(nPeriods)
            Median = np.zeros(nPeriods)
            Mean = np.zeros(nPeriods)
            Min = np.zeros(nPeriods)
            Max = np.zeros(nPeriods)
            
            for i in range(nPeriods):
                sr = seriesDfCopy[targetcol].iloc[ i - (w-1) : i + 1]
                
                NonzeroMeanLog[i] = np.log(sr[sr!=0].mean())
                Median[i] = sr.median()
                Mean[i] = sr.mean()
                Min[i] = sr.min()
                Max[i] = sr.max()
                
            seriesDfCopy['nonzero_mean_log_l' + str(w)] = NonzeroMeanLog
            seriesDfCopy['median_l' + str(w)] = Median
            seriesDfCopy['mean_l' + str(w)] = Mean
            seriesDfCopy['max_l' + str(w)] = Max
            seriesDfCopy['min_l' + str(w)] = Min
            
            container.append(seriesDfCopy.copy())

    data = pd.concat(container)
    data = data.groupby([dtcol,multiseries_id,targetcol]).sum().reset_index(drop=False)
    del container
    gc.collect()
    
    
    return data
