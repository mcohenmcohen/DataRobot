# Code Snippets and Helper Functions
These are code snippets intended for general purpose and are not use-case specific. 

## How to Use
Assuming you are already in the habit of cloning the [data-science-scripts](https://github.com/datarobot/data-science-scripts) Github repo to your laptop, all you need to do is append this directory path to python's `sys.path`.

There are several ways to do this.

On a temporary basis, you can add something like this to your imports section of your code:
```python
import pwd, sys
if pwd.getpwuid(os.getuid()).pw_name == 'josh.berry':
    sys.path.append("/Users/josh.berry/_company/data-science-scripts/joshberry/joshberry/")
    from joshberry.utils import *
    from joshberry.feats import *
```

Notice that the if statement will check if you are infact yourself, and conditionally adds that path. This helps in case you are sending this notebook to another individual.

This is currently how I do it (I use notebook templates so I don't have to type this in every time).

Another easymethod is to create a `my-paths.pth` file (as described [here](http://docs.python.org/library/site.html)). This is just a file with the extension `.pth` that you put into your system `site-packages` directory. You can add a different directory for each line. This might be the best solution if you're importing multiple custom code paths from other coworkers' Githubs. From python, run `import site; site.getsitepackages()` in order to see the path of site-packages where you would create the `.pth` file.

You could also use the PYTHONPATH environment variable, which is like the system `PATH` but contains directories that will be added to `sys.path`. See [documentation](http://docs.python.org/tutorial/modules.html#the-module-search-path).

----------------
## Contents
I am currently only using two modules: `feats` and `utils`

### Utils module
This module is intended to be utilities only. These are general purpose functions which I commonly use and are not specific to predictive modeling.

#### **`reduce_mem_usage`**(df, aggressive=`False`, verbose=`True`)

This function returns a `DataFrame` which has datatypes that are optimized to take up less memory. There is an `aggressive` flag which can reduce memory footprint even more -- however it can cause undesirable effects with floats where $12.99 might show up as $12.9889899989. This is probably due to the way python deals with floats. The default setting of `aggressive = False` will convert floats differently so that they retain extra precision to avoid that side effect.

Example Usage:

```python
indata = pd.read_csv(Path(data_dir) / FILENAME)
```
```python
indata = reduce_mem_usage(indata)
```
```
Mem. usage decreased to  1.96 MB (24.3% reduction)
```

#### **`combine_to_list`**(inlist, joiner)

Combines a list of items and returns a list with a custom separator

    Parameters:
            inlist (list): a list of items
            joiner (str) : a custom separator for the list
    Returns:
            list

#### **`chunks`**(l, n, randomized=`True`)

Chunks a long list into pieces of size n

    Parameters:
            l (list): a list
            n (int) : size n for max size of each chunk
    Returns:
            list of chunks (list of lists)

#### **`xldate_as_datetime`**(xldate, datemode)

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

#### **`xldatestring_as_datetime`**(df, datecols)
    Fixes the y/m/dd excel string date and makes it a real date

    Parameters:
            df (DataFrame)  : original dataframe
            datecols (list) : a list of date column names (str)
    Returns:
            DataFrame

Example with LendingClub:
```python
FILENAME = '10K_Lending_Club_Loans.csv'
```
```python
df = pd.read_csv(Path(data_dir) / FILENAME, parse_dates=True, encoding='latin1')
```
```python
df = df[['loan_amnt','term','earliest_cr_line','is_bad']][:10]
```
```
df
```
| loan_amnt |   term    | earliest_cr_line | is_bad |
|-----------|-----------|------------------|--------|
|     14000 | 60 months | 12/1/00          |      1 |
|      8750 | 36 months | 1/1/91           |      0 |
|     15000 | 36 months | 3/1/97           |      0 |
|      8875 | 36 months | 12/1/98          |      1 |
|     15000 | 60 months | 5/1/95           |      0 |
|     16000 | 36 months | 11/1/97          |      0 |
|      5000 | 36 months | 3/1/02           |      0 |
|      6000 | 36 months | 12/1/94          |      0 |
|      8000 | 36 months | 2/1/06           |      0 |
|     10000 | 36 months | 6/1/90           |      0 |

 
 Notice the atrocious date format (thanks, Excel!). Here's the function that fixes that:

```python
dtcols = ['earliest_cr_line']

df = xldatestring_as_datetime(df, dtcols)
```
```
----------
ORIGINAL: 12/1/00
MODIFIED: 2000-12-01 00:00:00
```
Notice we get a single example printed into the output so that we can analyze whether it worked or not (I still recommend checking closer than just 1 example!)

Now, it's fixed:
```
df
```
| loan_amnt |   term    | earliest_cr_line | is_bad |
|-----------|-----------|------------------|--------|
|     14000 | 60 months | 2000-12-01 0:00  |      1 |
|      8750 | 36 months | 1991-01-01 0:00  |      0 |
|     15000 | 36 months | 1997-03-01 0:00  |      0 |
|      8875 | 36 months | 1998-12-01 0:00  |      1 |
|     15000 | 60 months | 1995-05-01 0:00  |      0 |
|     16000 | 36 months | 1997-11-01 0:00  |      0 |
|      5000 | 36 months | 2002-03-01 0:00  |      0 |
|      6000 | 36 months | 1994-12-01 0:00  |      0 |
|      8000 | 36 months | 2006-02-01 0:00  |      0 |
|     10000 | 36 months | 1990-06-01 0:00  |      0 |




---------------------
### Feats module
This module is intended to be used related to feature engineering and data manipulation.

#### **`calculate_woe_iv`**(dataset, feature, target)

    Calculates the Information Value of a specific variable against a binary target.
    Note that this only works for categorical values because we have not bothered to 
    incorporate any binning strategies for numerics

            Parameters:
                    dataset (DataFrame) : DataFrame
                    feature (str)       : name of the feature to analyze
                    target (str)        : name of the target feature
            Returns:
                    Information Value (float)

#### **`add_partition`**(df, k=5, holdout_size=0.20, target_variable_name='target', holdout_indicator_name='holdout_ind')

    Modifies a dataset to include a partition indicator using KFold stratified sampling. For this reason
    it is meant for a binary target only.

            Parameters:
                    df (DataFrame)               : DataFrame
                    k (int)                      : number of training partitions (excluding holdout)
                    holdout_size (float)         : percentage of holdout
                    target_variable_name (str)   : name of target variable (binary only)
                    holdout_indicator_name (str) : name of the holdout indicator to be created
            Returns:
                    DataFrame with holdout indicator added

#### **`add_business_day_cols`**(df, datecols)

    Modifies a dataset to include new columns that describe the date in terms of business days remaining and
    elapsed in the month

            Parameters:
                    df (DataFrame)               : DataFrame
                    datecols (list of str)       : number of datecolumns for which to perform the operation

            Returns:
                    DataFrame with new columns added

Example:
```python
# let's make a fake dataframe that contains a date
df = pd.DataFrame({"loan_amnt":{"2232":14000,"9055":8750,"8711":15000,"9262":8875,"2622":15000,"6131":16000,"5813":5000,"6465":6000,"5530":8000,"7675":10000},"term":{"2232":" 60 months","9055":" 36 months","8711":" 36 months","9262":" 36 months","2622":" 60 months","6131":" 36 months","5813":" 36 months","6465":" 36 months","5530":" 36 months","7675":" 36 months"},"earliest_cr_line":{"2232":"2000-12-01 00:00:00","9055":"1991-01-01 00:00:00","8711":"1997-03-01 00:00:00","9262":"1998-12-01 00:00:00","2622":"1995-05-01 00:00:00","6131":"1997-11-01 00:00:00","5813":"2002-03-01 00:00:00","6465":"1994-12-01 00:00:00","5530":"2006-02-01 00:00:00","7675":"1990-06-01 00:00:00"},"is_bad":{"2232":1,"9055":0,"8711":0,"9262":1,"2622":0,"6131":0,"5813":0,"6465":0,"5530":0,"7675":0}})
```
```
df
```
| loan_amnt |   term    | earliest_cr_line | is_bad |
|-----------|-----------|------------------|--------|
|     14000 | 60 months | 2000-12-01 0:00  |      1 |
|      8750 | 36 months | 1991-01-01 0:00  |      0 |
|     15000 | 36 months | 1997-03-01 0:00  |      0 |
|      8875 | 36 months | 1998-12-01 0:00  |      1 |
|     15000 | 60 months | 1995-05-01 0:00  |      0 |
|     16000 | 36 months | 1997-11-01 0:00  |      0 |
|      5000 | 36 months | 2002-03-01 0:00  |      0 |
|      6000 | 36 months | 1994-12-01 0:00  |      0 |
|      8000 | 36 months | 2006-02-01 0:00  |      0 |
|     10000 | 36 months | 1990-06-01 0:00  |      0 |



We should make sure our columns are a date type before we proceed:
```python
dtcols = ['earliest_cr_line']
```
```
for c in dtcols:
    df[c] = pd.to_datetime(df[c])
```

Now we can add business-day metrics about the date

```python
result = add_business_day_cols(df, dtcols)
```
```
result
```
| loan_amnt |   term    | earliest_cr_line | is_bad | num_sats_exist_in_mo_earliest_cr_line | num_sats_elapsed_in_mo_earliest_cr_line | num_sats_remain_in_mo_earliest_cr_line | num_bizdays_exist_in_mo_earliest_cr_line | num_bizdays_elapsed_in_mo_earliest_cr_line | num_bizdays_remain_in_mo_earliest_cr_line |
|-----------|-----------|------------------|--------|---------------------------------------|-----------------------------------------|----------------------------------------|------------------------------------------|--------------------------------------------|-------------------------------------------|
|     14000 | 60 months | 2000-12-01       |      1 |                                     5 |                                       0 |                                      5 |                                       21 |                                          0 |                                        21 |
|      8750 | 36 months | 1991-01-01       |      0 |                                     4 |                                       0 |                                      4 |                                       23 |                                          0 |                                        23 |
|     15000 | 36 months | 1997-03-01       |      0 |                                     5 |                                       0 |                                      5 |                                       21 |                                          0 |                                        21 |
|      8875 | 36 months | 1998-12-01       |      1 |                                     4 |                                       0 |                                      4 |                                       23 |                                          0 |                                        23 |
|     15000 | 60 months | 1995-05-01       |      0 |                                     4 |                                       0 |                                      4 |                                       23 |                                          0 |                                        23 |
|     16000 | 36 months | 1997-11-01       |      0 |                                     5 |                                       0 |                                      5 |                                       20 |                                          0 |                                        20 |
|      5000 | 36 months | 2002-03-01       |      0 |                                     5 |                                       0 |                                      5 |                                       21 |                                          0 |                                        21 |
|      6000 | 36 months | 1994-12-01       |      0 |                                     5 |                                       0 |                                      5 |                                       22 |                                          0 |                                        22 |
|      8000 | 36 months | 2006-02-01       |      0 |                                     4 |                                       0 |                                      4 |                                       20 |                                          0 |                                        20 |
|     10000 | 36 months | 1990-06-01       |      0 |                                     5 |                                       0 |                                      5 |                                       21 |                                          0 |                                        21 |



#### **`add_calendar_cols`**(dfMain, dfEvents, date_col_list, eventname_col='Name', eventdate_col='Date')

    References a calendar file and adds variables to the main dataset of how far away each "event" is

            Parameters:
                    dfMain  (DataFrame)  : main dataset with 1 or more date columns to reference
                    dfEvents (DataFrame) : calendar dataset with 2 columns, 1 for date and 1 for name of the event
                    date_col_list (str)  : list of reference dates from the main dataset
                    eventname_col (str)  : name of the column for DATE in the calendar dataset
                    eventdate_col (str)  : name of the column for EVENT NAME in calendar dataset
            Returns:
                    DataFrame

Example of usage:

We'll create a fake dataframe, along with a fake "calendar table" which contains dates and (in this case) Holidays.
```python
df = pd.DataFrame({"loan_amnt":{"2232":14000,"9055":8750,"8711":15000,"9262":8875,"2622":15000,"6131":16000,"5813":5000,"6465":6000,"5530":8000,"7675":10000},"term":{"2232":" 60 months","9055":" 36 months","8711":" 36 months","9262":" 36 months","2622":" 60 months","6131":" 36 months","5813":" 36 months","6465":" 36 months","5530":" 36 months","7675":" 36 months"},"earliest_cr_line":{"2232":"2000-12-01 00:00:00","9055":"2000-01-01 00:00:00","8711":"2000-03-01 00:00:00","9262":"2000-12-01 00:00:00","2622":"2000-05-01 00:00:00","6131":"2000-11-01 00:00:00","5813":"2000-03-01 00:00:00","6465":"2000-12-01 00:00:00","5530":"2000-02-01 00:00:00","7675":"2000-06-01 00:00:00"},"is_bad":{"2232":1,"9055":0,"8711":0,"9262":1,"2622":0,"6131":0,"5813":0,"6465":0,"5530":0,"7675":0}})

```

| loan_amnt |   term    | earliest_cr_line | is_bad |
|-----------|-----------|------------------|--------|
|     14000 | 60 months | 2000-12-01 0:00  |      1 |
|      8750 | 36 months | 2000-01-01 0:00  |      0 |
|     15000 | 36 months | 2000-03-01 0:00  |      0 |
|      8875 | 36 months | 2000-12-01 0:00  |      1 |
|     15000 | 60 months | 2000-05-01 0:00  |      0 |
|     16000 | 36 months | 2000-11-01 0:00  |      0 |
|      5000 | 36 months | 2000-03-01 0:00  |      0 |
|      6000 | 36 months | 2000-12-01 0:00  |      0 |
|      8000 | 36 months | 2000-02-01 0:00  |      0 |
|     10000 | 36 months | 2000-06-01 0:00  |      0 |


```python
dfCal = pd.DataFrame({"holiday_dt":{"0":"1999-04-04","1":"2000-04-23","2":"2001-04-15","3":"1999-01-01","4":"1999-01-18","5":"1999-05-31","6":"1999-07-04","7":"1999-09-06","8":"1999-10-11","9":"1999-11-11","10":"1999-11-25","11":"1999-11-26","12":"1999-12-25","13":"1999-12-18","14":"2000-01-01","15":"2000-01-17","16":"2000-05-29","17":"2000-07-04","18":"2000-09-04","19":"2000-10-09","20":"2000-11-11","21":"2000-11-23","22":"2000-11-24","23":"2000-12-25","24":"2000-12-23","25":"2001-01-01","26":"2001-01-15","27":"2001-05-28","28":"2001-07-04","29":"2001-09-03","30":"2001-10-08","31":"2001-11-11","32":"2001-11-22","33":"2001-11-23","34":"2001-12-25","35":"2001-12-22"},"holiday":{"0":"Easter","1":"Easter","2":"Easter","3":"New Year\'s Day","4":"Martin Luther King, Jr. Day","5":"Memorial Day","6":"Independence Day","7":"Labor Day","8":"Columbus Day","9":"Veterans Day","10":"Thanksgiving","11":"Black Friday","12":"Christmas Day","13":"Sat Before X-max","14":"New Year\'s Day","15":"Martin Luther King, Jr. Day","16":"Memorial Day","17":"Independence Day","18":"Labor Day","19":"Columbus Day","20":"Veterans Day","21":"Thanksgiving","22":"Black Friday","23":"Christmas Day","24":"Sat Before X-max","25":"New Year\'s Day","26":"Martin Luther King, Jr. Day","27":"Memorial Day","28":"Independence Day","29":"Labor Day","30":"Columbus Day","31":"Veterans Day","32":"Thanksgiving","33":"Black Friday","34":"Christmas Day","35":"Sat Before X-max"}})
```
| holiday_dt |           holiday           |
|------------|-----------------------------|
| 1999-04-04 | Easter                      |
| 2000-04-23 | Easter                      |
| 2001-04-15 | Easter                      |
| 1999-01-01 | New Year's Day              |
| 1999-01-18 | Martin Luther King, Jr. Day |
| ...        | ...                         |
| 2001-11-11 | Veterans Day                |
| 2001-11-22 | Thanksgiving                |
| 2001-11-23 | Black Friday                |
| 2001-12-25 | Christmas Day               |
| 2001-12-22 | Sat Before X-max            |

Now, we can use the function to create `days_since` and `days until` variables for each unique holiday.

We must pass both dataframes, along with a list of date columns (this function will loop). 

We assume the dfCal file has headers called `Date` and `Name`, but if they don't -- then you have to pass in the variable names explicitly like in this example below:
```python
result = add_calendar_cols(df, dfCal, ['earliest_cr_line'], eventname_col='holiday', eventdate_col='holiday_dt')
```
```
result
```


| loan_amnt |   term    | earliest_cr_line | is_bad | days_until_next_event | days_since_last_event | earliest_cr_line_days_until_Easter | earliest_cr_line_days_since_Easter | earliest_cr_line_days_until_New Year's Day | earliest_cr_line_days_since_New Year's Day | earliest_cr_line_days_until_Martin Luther King, Jr. Day | earliest_cr_line_days_since_Martin Luther King, Jr. Day | earliest_cr_line_days_until_Memorial Day | earliest_cr_line_days_since_Memorial Day | earliest_cr_line_days_until_Independence Day | earliest_cr_line_days_since_Independence Day | earliest_cr_line_days_until_Labor Day | earliest_cr_line_days_since_Labor Day | earliest_cr_line_days_until_Columbus Day | earliest_cr_line_days_since_Columbus Day | earliest_cr_line_days_until_Veterans Day | earliest_cr_line_days_since_Veterans Day | earliest_cr_line_days_until_Thanksgiving | earliest_cr_line_days_since_Thanksgiving | earliest_cr_line_days_until_Black Friday | earliest_cr_line_days_since_Black Friday | earliest_cr_line_days_until_Christmas Day | earliest_cr_line_days_since_Christmas Day | earliest_cr_line_days_until_Sat Before X-max | earliest_cr_line_days_since_Sat Before X-max |
|-----------|-----------|------------------|--------|-----------------------|-----------------------|------------------------------------|------------------------------------|--------------------------------------------|--------------------------------------------|---------------------------------------------------------|---------------------------------------------------------|------------------------------------------|------------------------------------------|----------------------------------------------|----------------------------------------------|---------------------------------------|---------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|-------------------------------------------|-------------------------------------------|----------------------------------------------|----------------------------------------------|
|     14000 | 60 months | 2000-12-01       |      1 |                    22 |                     7 |                                135 |                                222 |                                         31 |                                        335 |                                                      45 |                                                     319 |                                      178 |                                      186 |                                          215 |                                          150 |                                   276 |                                    88 |                                      311 |                                       53 |                                      345 |                                       20 |                                      356 |                                        8 |                                      357 |                                        7 |                                        24 |                                       342 |                                           22 |                                          349 |
|      8750 | 36 months | 2000-01-01       |      0 |                     0 |                     0 |                                113 |                                272 |                                          0 |                                          0 |                                                      16 |                                                     348 |                                      149 |                                      215 |                                          185 |                                          181 |                                   247 |                                   117 |                                      282 |                                       82 |                                      315 |                                       51 |                                      327 |                                       37 |                                      328 |                                       36 |                                       359 |                                         7 |                                          357 |                                           14 |
|     15000 | 36 months | 2000-03-01       |      0 |                    53 |                    44 |                                 53 |                                332 |                                        306 |                                         60 |                                                     320 |                                                      44 |                                       89 |                                      275 |                                          125 |                                          241 |                                   187 |                                   177 |                                      222 |                                      142 |                                      255 |                                      111 |                                      267 |                                       97 |                                      268 |                                       96 |                                       299 |                                        67 |                                          297 |                                           74 |
|      8875 | 36 months | 2000-12-01       |      1 |                    22 |                     7 |                                135 |                                222 |                                         31 |                                        335 |                                                      45 |                                                     319 |                                      178 |                                      186 |                                          215 |                                          150 |                                   276 |                                    88 |                                      311 |                                       53 |                                      345 |                                       20 |                                      356 |                                        8 |                                      357 |                                        7 |                                        24 |                                       342 |                                           22 |                                          349 |
|     15000 | 60 months | 2000-05-01       |      0 |                    28 |                     8 |                                349 |                                  8 |                                        245 |                                        121 |                                                     259 |                                                     105 |                                       28 |                                      336 |                                           64 |                                          302 |                                   126 |                                   238 |                                      161 |                                      203 |                                      194 |                                      172 |                                      206 |                                      158 |                                      207 |                                      157 |                                       238 |                                       128 |                                          236 |                                          135 |
|     16000 | 36 months | 2000-11-01       |      0 |                    10 |                    23 |                                165 |                                192 |                                         61 |                                        305 |                                                      75 |                                                     289 |                                      208 |                                      156 |                                          245 |                                          120 |                                   306 |                                    58 |                                      341 |                                       23 |                                       10 |                                      356 |                                       22 |                                      342 |                                       23 |                                      341 |                                        54 |                                       312 |                                           52 |                                          319 |
|      5000 | 36 months | 2000-03-01       |      0 |                    53 |                    44 |                                 53 |                                332 |                                        306 |                                         60 |                                                     320 |                                                      44 |                                       89 |                                      275 |                                          125 |                                          241 |                                   187 |                                   177 |                                      222 |                                      142 |                                      255 |                                      111 |                                      267 |                                       97 |                                      268 |                                       96 |                                       299 |                                        67 |                                          297 |                                           74 |
|      6000 | 36 months | 2000-12-01       |      0 |                    22 |                     7 |                                135 |                                222 |                                         31 |                                        335 |                                                      45 |                                                     319 |                                      178 |                                      186 |                                          215 |                                          150 |                                   276 |                                    88 |                                      311 |                                       53 |                                      345 |                                       20 |                                      356 |                                        8 |                                      357 |                                        7 |                                        24 |                                       342 |                                           22 |                                          349 |
|      8000 | 36 months | 2000-02-01       |      0 |                    82 |                    15 |                                 82 |                                303 |                                        335 |                                         31 |                                                     349 |                                                      15 |                                      118 |                                      246 |                                          154 |                                          212 |                                   216 |                                   148 |                                      251 |                                      113 |                                      284 |                                       82 |                                      296 |                                       68 |                                      297 |                                       67 |                                       328 |                                        38 |                                          326 |                                           45 |
|     10000 | 36 months | 2000-06-01       |      0 |                    33 |                     3 |                                318 |                                 39 |                                        214 |                                        152 |                                                     228 |                                                     136 |                                      361 |                                        3 |                                           33 |                                          333 |                                    95 |                                   269 |                                      130 |                                      234 |                                      163 |                                      203 |                                      175 |                                      189 |                                      176 |                                      188 |                                       207 |                                       159 |                                          205 |                                          166 |

#### **`sax`**(df, dtcol, groupbycols, targetcol, n_bins=5, strategy='quantile')

        Parameters:
                df  (DataFrame)    : main time-series ready dataset
                dtcol (str)        : name of column with date
                groupbycols (list) : list of group by columns to represent multiseries
                targetcol (str)    : name of the column that is the target
                n_bins (int)       : number of bins for sax to recognize. must be 2 <= n <= 26
                strategy (str)     : 'quantile', 'normal' - depending on distrubution of target
        Returns:
                DataFrame
