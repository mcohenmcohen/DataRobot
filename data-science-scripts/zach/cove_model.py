### Python3
from keras.models import load_model
from os import chdir
from os.path import expanduser

chdir(expanduser('~/source/CoVe/'))
cove_model = load_model('Keras_CoVe.h5')

cove_model.save_weights('Keras_CoVe_weights.h5')                                                                                                                                                                                                                         
json_string = cove_model.to_json()

with open('Keras_CoVe.json', 'w') as outfile:
    outfile.write(json_string)

### Python2
### NOPE DOESNT WORK
from keras.models import model_from_json
from os import chdir
from os.path import expanduser

with open('Keras_CoVe_json.json', 'r') as infile:
    json_string = infile.read()

py2_model = model_from_json(json_string)
py2_model.load_weights('Keras_CoVe_weights.h5')

py2_model.save('Keras_CoVe_py2.h5')
py2_model_check = load_model('Keras_CoVe_py2.h5')



from sklearn.feature_extraction import DictVectorizer, FeatureHasher
import json
v = DictVectorizer(sparse=True)
count_json = ['{"foo": 1, "bar": 2}', '{"foo": 3, "baz": 1}']
count_dict = [json.loads(x) for x in count_json]
v.fit_transform(count_json).toarray()
