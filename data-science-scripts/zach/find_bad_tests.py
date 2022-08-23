import fnmatch
import imp
import inspect
import os
import sys
import unittest


def extract_classes_from_module(module_to_look_at):
    """
    Find all classes in a module that are not unittest.TestCase and don't start with Test

    These are test classes that pytest will not run

    TODO: Skip imports
    """
    all_classes_not_imported = [m[1] for m in inspect.getmembers(module_to_look_at, inspect.isclass)
                                if m[1].__module__ == module_to_look_at.__name__]

    for not_imported_class in all_classes_not_imported:
        if isinstance(not_imported_class, unittest.TestCase):
            continue
        if not_imported_class.__name__.startswith('Test') and \
                not_imported_class.__name__.startswith('test'):
            continue
        yield not_imported_class


def should_class_be_test_class(class_to_look_at):
    """
    Look for functions that start with 'test' in a class.  If found, return True
    This means that the class containing this function should probably be a test class.
    """
    for name, obj in class_to_look_at.__dict__.items():
        if name.startswith('test') and type(obj) == type(lambda: 1):
            return True
    return False


def check_model_doesnt_have_bad_test_classes(module_to_look_at):
    """
    Combine the 2 functions above:
    - loop through all potential bad test classes in a module
    - assert those bad modules don't have potential tests
    """
    classes_gen = extract_classes_from_module(module_to_look_at)
    for bad_class in classes_gen:
        assert not should_class_be_test_class(bad_class), bad_class

sys.path.append(os.path.dirname('')+'/ModelingMachine')
for root, dirnames_, filenames in os.walk('tests/tasks2/'):
    for filename in fnmatch.filter(filenames, '*.py'):
        if filename == '__init__.py':
            continue
        filename = os.path.join(root, filename)
        try:
            mod = imp.load_source('test_target', filename)
            check_model_doesnt_have_bad_test_classes(mod)
            sys.modules.pop('test_target')
        except AssertionError as err:
            print('I found a bad test case in', filename, str(err))
        except Exception as err:
            pass
            #print('Failed to handle', filename, str(err))