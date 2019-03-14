import argparse
from collections import OrderedDict
import csv
from itertools import groupby
import os

import pandas as pd

parser = argparse.ArgumentParser(description='Convert long format of reason codes to ui format')
parser.add_argument('input_long_format_file',
                    help='filename of long format')
parser.add_argument('output_filename_ui_format',
                    help='UI format, must not already exist')
args = parser.parse_args()

if os.path.exists(args.output_filename_ui_format):
    sys.stderr.write('*** ERROR: Will not overwrite "{:}", aborting\n'.format(args.output_filename_ui_format))
    sys.exit()

output_lines = []
with open(args.input_long_format_file) as infile:
    reader = csv.DictReader(infile)
    for row_id, group in groupby(reader, key=lambda line: line['row_id']):
        lines = list(group)
        this_row = OrderedDict()
        this_row['row_id'] = row_id
        this_row['Prediction'] = lines[0]['prediction']
        for feature_number, line in enumerate(lines, 1):
            this_row['Reason {:} Strength'.format(feature_number)] = line['qualitativeStrength']
            this_row['Reason {:} Feature'.format(feature_number)] = line['feature']
            this_row['Reason {:} Value'.format(feature_number)] = line['featureValue']
        output_lines.append(this_row)
pd.DataFrame(output_lines).to_csv(args.output_filename_ui_format, index=False)
