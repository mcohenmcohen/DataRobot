# http://trafaret.readthedocs.io/en/latest/intro.html

import trafaret as t
import json

GENERIC_DICT = t.Dict().allow_extra('*')

class Blueprint(object):
  @t.guard(
      task_map_key=t.String(
          allow_blank=False,
          min_length=2, #Task names must 2 or more letters
          max_length=32, #Task names must 32 or fewer letters
          regex='^[A-ZA-Z0-9]+$' #Task names must be all caps + numbers
      ),
      arguments=GENERIC_DICT
  )
  def __init__(self, task_map_key, arguments):
    self.task_map_key = task_map_key
    self.arguments = arguments

  def to_json(self):
      return json.dumps(self.__dict__)

  @classmethod
  def from_json(cls, json_str):
    return cls(**json.loads(json_str))

  def __str__(self):
    return self.to_json()

# Make blueprint
model1 = Blueprint(task_map_key="STATS", arguments={"window":int(5), "method":"mean"})

# Serialize to json and deserialize to a new BP
json_str1 = model1.to_json()
model2 = Blueprint.from_json(json_str1)
json_str2 = model2.to_json()

# Confirm the 2 BPs are identical
print json_str1
print json_str2
print json_str1 == json_str2
