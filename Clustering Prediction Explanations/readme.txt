Slack: #clusterin_pred_expl

SierraLeone_Water_data_small_20181211.csv
- target = "status binary" ; leakage in "pay"

pay and status_id are leakage

age is good
status is leakage
but pay and status would be interesting later, in the dashboard as filters, which is why I left them in there
'adm1' - state, 'adm2',- cirty 'country_name', 'install_year', 'installer', 'lat_deg',
      'lon_deg',
'management' - who manages it's maintenance, if at all
, 'pay', - does the community pay for it (leakage)
'report_date', -date of report, there were two major data collections in 2012 and 2015
'source', - who collected it
'status', - raw text notes about functionality. not used in modeling, rich source of info
      'status_id', - short form of "working/ not working/ maybe working"
'water_source', - where water comes from
'water_tech', - pump, bucket, etc - technology of the well
'wpdx_id',
***derived below***
'new_report_date',- cleaner date in pandas
      'new_install_year', - cleaner year waterpoint was built
'age_well', - report date - install year
'age_well_days', - age in days
'status_binary',- binary working or not,

      'time_since_measurement', - report date - today
'time_since_meas_years',
'fuzzy_water_source', - fuzzy matching to create categories of sources
      'fuzzy_water_tech'],- same for tech (edited) 
