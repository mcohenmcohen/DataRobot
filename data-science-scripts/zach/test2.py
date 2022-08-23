from __future__ import division
from __future__ import unicode_literals

import os
import itertools
import unittest
import random

import numpy as np
import pandas
import pandas as pd
from pickle import dumps, loads
from scipy.stats import norm
from binascii import hexlify
import pytest

from ModelingMachine.engine.modeling_data import create_modeling_data, create_task_data
from ModelingMachine.engine.modeling_data import subset_data
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.tasks2.text_converters import SingleColumnText, SingleColumnText2
from ModelingMachine.engine.tasks2.text_converters import TfIdf2, TfIdf3
from ModelingMachine.engine.tasks2.text_converters import VWConverter, VWConverter2
from ModelingMachine.engine.tasks2.nlp import char_preprocessor
from ModelingMachine.engine.tasks2.stopwords import SUPPORTED_STOPWORD_LANGUAGES
from ModelingMachine.engine.tasks2.nlp import SUPPORTED_SNOWBALL_LANGUAGES
from ModelingMachine.engine.tasks2.nlp import char_preprocessor
from ModelingMachine.engine.tasks2.nlp import Preprocessor
from ModelingMachine.engine.tasks2.japNLP import TinySegmenter
from ModelingMachine.engine.modeling_data import subset_data
from ModelingMachine.engine.modeling_data import create_modeling_data, create_task_data
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.code_generation.code_generator import CodeGenerator
from tests.functional.synth_dataset import RichSeries
from common.services.transparent_models import Transformation, ModelParameter
from common.exceptions import VersionUnsupportedError
from common.enum import TaskMethod
from tests.tasks2.base_task_test import BaseTaskTest

X = np.array([
    ['a1 a2 a3 a4 a5'],
    ['a4 a5 a2 a4'],
    ['a1 a2'],
    ['a3 a4'],
    ['a4 a5 a1 a2 a3 a4'],
])
X = np.array([
    ['1 2 3 4 5'],
    ['4 5 2 4'],
    ['1 2'],
    ['3 4'],
    ['4 5 1 2 3 4'],
])

y = np.random.choice([0, 1], X.shape[0])  # pylint: disable=no-member
modeling_data = create_modeling_data(X=X, Y=y, target=y, colnames=['seq'],
                                     row_index=np.arange(X.shape[0]))
task_data = create_task_data(cv_method='RandomCV')

# this will error
task = TfIdf3('a=word;d2=0.5')
out = task.fit_transform(modeling_data, task_data)
out_X = out.X.toarray()
out_cols = out.colnames
print(out_cols)
print(out_X)

# Make sure the output looks correct
assert out_X.shape[0] == X.shape[0]
assert out_X.shape[1] == 5
assert out_cols == ['seq- ', 'seq-1', 'seq-2', 'seq-3', 'seq-4', 'seq-5']

# make sure the vectorizers now have the correct settings
assert task.vec['sequence'].min_df == 1
assert task.vec['sequence'].max_df == 1.0
assert task.vec['sequence'].stop_words == None
assert task.vec['sequence'].preprocessor == None
assert task.vec['sequence'].token_pattern == '\\b\\w+\\b'
