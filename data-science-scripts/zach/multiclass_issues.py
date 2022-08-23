
###########################################################################
# Fit a TF vertex to 100 classes
###########################################################################

from sklearn.datasets import make_classification
from ModelingMachine.engine.modeling_data import create_modeling_data, create_task_data
from ModelingMachine.engine.vertex import VertexV2
from ModelingMachine.engine.response import Response
import pandas as pd
import pickle

class_count = 100
X, y = make_classification(
	random_state=123,
	n_samples=1000,
	n_classes=class_count,
	n_features=50,
	n_informative=40,
	n_redundant=5,
	n_repeated=5
)
modeling_data = create_modeling_data(
	X=X,
	Y=Response.from_array(y, target_type='Multiclass'),
	row_index=pd.Series(range(X.shape[0])),
	colnames=range(X.shape[1]),
	user_partition=pd.Series(range(X.shape[0])))
task_data = create_task_data(
	inputs=['N', 'C', 'T'], class_count=class_count
)
# add target data
target = pd.Series(data=modeling_data.Y, index=range(modeling_data.Y.shape[0]))
modeling_data = modeling_data._replace(target=target)

for task in ['PNI2 dtype=float32', 'RST dtype=float32']:
	vertex = VertexV2([task], 'id')
	modeling_data = vertex.fit_transform(modeling_data, task_data).data

vertex = VertexV2(['TFNNC ni=2000;bs=128;es=full;h=[25, 256];p=0.5;ew=1;ep=10;t_light=1;t_n=1;t_f=0.15;t_a=3;t_sp=1;t_sf=1;t_es_init=5;t_m=LogLoss'], 'id')
new_vertex = deepcopy(vertex)

fit_vertex = vertex.fit(modeling_data, task_data)
pred  = vertex.predict(modeling_data, task_data)

fit_vertex.save("/home/ubuntu/vertex2.pkl")
new_vertex = pickle.load(open("/home/ubuntu/vertex2.pkl", 'rb'))


###########################################################################
# Find numpy arrays in the vertex
###########################################################################

import collections
import numpy as np
from six import string_types
def find_np_arrays(data):
    found_arrays = []
    objects_examined = collections.Counter()
    key_stack = []
    def _handle_dict(d):
        for k, v in d.items():
            # print('key', k)
            skip_keys = [
                '__builtins__',
                'path_importer_cache',
                'categorical_columns',
                'stdout',
                'bs4',
                'six.moves',
                'sys',
                'warnings',
                'shell',
                'displayhook'
            ]
            if k.startswith('__') and k.endswith('__'):
                # print('Skipping {}'.format(k))
                continue
            # if k in __builtins__:
            #     print('Skipping builtin {}'.format(k))
            #     continue
            if k in skip_keys:
                # print('Skipping {}'.format(k))
                continue
            key_stack.append(k)
            _handle(v)
            key_stack.pop()
        return d
    def _handle_sequence(l):
        for idx, v in enumerate(l):
            key_stack.append(idx)
            _handle(v)
            key_stack.pop()
        return l
    def _handle_np_array(a):
        current_stack = key_stack[:]
        if a not in found_arrays:
            found_arrays.append((current_stack, a, a.shape))
    def _handle(item):
        is_dict = isinstance(item, dict)
        is_sequence = isinstance(item, collections.Sequence)
        is_string = isinstance(item, string_types)
        is_numpy_array = isinstance(item, (np.ndarray, np.generic)) or type(item).__module__ == np.__name__
        if is_dict:
            _handle_dict(item)
        elif is_sequence and not is_string:
            _handle_sequence(item)
        elif is_numpy_array:
            _handle_np_array(item)
        else:
            # print(item)
            key_stack.append('__dict__')
            try:
                if item.__dict__:
                    item_id = id(item)
                    objects_examined[item_id] += 1
                    if objects_examined[item_id] < 3:
                        _handle_dict(item.__dict__)
            except:
                pass
            finally:
                key_stack.pop()
    _handle(data)
    return found_arrays

def display_path(key_path):
    display_str = 'classifier'
    for item in key_path:
        if item == '__dict__':
            display_str += '.__dict__'
        else:
            if isinstance(item, int):
                display_str += '[' + str(item) + ']'
            else:
                display_str += '[\'' + item + '\']'
    return display_str

def display(found_arrays):
    for fa in found_arrays:
        print(display_path(fa[0]))
        print(fa[1])


found_results = find_np_arrays(vertex.steps[0][0].model)  # vertex.steps[0][0].model.model._data_feeder.X.shape
path_parts, array, shape = found_results[0]
display_path(path_parts)
