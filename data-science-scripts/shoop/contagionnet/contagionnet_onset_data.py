"""
Script by Shoop. This script assumes Python3 execution.
Comment out the c4d import lines if you are not running this in Cinema 4D Script Window.
"""

import c4d
import csv

from c4d import gui
from pathlib import Path

def main():
    time_data = {}
    
    # NOTE: replace raw_filepath with wherever you saved MODELED_POINTS.csv
    raw_filepath = "./MODELED_POINTS_FOR_VIDEO_RED-full-20201210 - MODELED_POINTS_FOR_VIDEO_RED-full-20201210.csv"
    file_path = Path(raw_filepath)
    
    # Importing data file, reading data into Python Dict
    with file_path.open() as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            # row is Dict with keys:
            # {'date','smsa_name','id','x','y','latitude','longitude',
            # 'z','death_interval', 'time_of_onset'}
            time_onset = row['time_of_onset']
            if time_onset not in time_data:
                time_data[time_onset] = []
            # Adding tuple of lat/long coords to time_of_onset list
            lat_and_long = (
                float(row['latitude']),
                float(row['longitude'])
            )
            time_data[time_onset].append(lat_and_long)

    # Sorting by time_of_onset
    sorted_time_data = dict(sorted(time_data.items()))

    # Getting list of lists for proper C4D Outputs (NOTE: not so efficient...)
    latitude_list = []
    longitude_list = []
    for time_data_values in sorted_time_data.values():
        time_latitudes = []
        time_longitudes = []
        for lat_and_long in time_data_values:
            time_latitudes.append(lat_and_long[0])
            time_longitudes.append(lat_and_long[1])
        latitude_list.append(time_latitudes)
        longitude_list.append(time_longitudes)

    # Outputs for C4D Python Node (ordered by time_of_onset)
    global Output1  # time_of_onset date
    global Output2  # latitude list
    global Output3  # longitude list

    # C4D Python output nodes will depend on Input1, such as time
    # Input1 = 8
    ctr = int(Input1)
    Output1 = list(sorted_time_data)[ctr]
    Output2 = latitude_list[ctr]
    Output3 = longitude_list[ctr]

    # Debugging
    # -----------------------------------
    # max_dots_date = ""
    # max_dots_num = 0
    # for key, vals in sorted_time_data.items():
    #     num_dots = len(vals)
    #     if num_dots > max_dots_num:
    #         max_dots_date = key
    #         max_dots_num = num_dots
    # print(max_dots_date)
    # print(max_dots_num)
    # print(len(list(sorted_time_data)))
    # import json
    # with open("time_data.json", 'w') as output_file:
    #     json.dump(sorted_time_data, output_file, indent=4)
    # print(Output1)
    # print(Output2)
    # print(Output3)


if __name__=="__main__":
    main()
