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



found_results = find_np_arrays(classifier)

path_parts, array, shape = found_results[2]
display_path(path_parts)
