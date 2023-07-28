import datetime
import logging
import time
import itertools
import copy
import hashlib
import base64

# Functions

def chunk_data(data, size):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in itertools.islice(it, size)}

def split_data(data, num_splits):
    out = {}
    for i, key in enumerate(data):
        if i % num_splits == 0:
            yield out
            out = {}
        out[key] = data[key]
    if out:
        yield out

def make_hash_sha256(o):
    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(o)).encode())
    nonce = base64.b64encode(hasher.digest()).decode()
    return int.from_bytes(base64.b64decode(nonce), 'big')

def make_hash(o):
  """
  Makes a hash from a dictionary, list, tuple or set to any level, that contains
  only other hashable types (including any lists, tuples, sets, and
  dictionaries).
  Taken from: https://stackoverflow.com/questions/5884066/hashing-a-dictionary
  """

  if isinstance(o, (set, tuple, list)):
    return tuple([make_hash(e) for e in o])    
  elif not isinstance(o, dict):
    return hash(o)
  new_o = copy.deepcopy(o)
  for k, v in new_o.items():
    new_o[k] = make_hash(v)
  return hash(tuple(frozenset(sorted(new_o.items()))))

def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k,make_hashable(v)) for k,v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o

# Classes

class DeltaTimeFormatter(logging.Formatter):
    def format(self, record):
        duration = datetime.datetime.utcfromtimestamp(record.relativeCreated / 1000)
        record.delta = duration.strftime("%H:%M:%S")
        return super().format(record)

class HashableDict(dict):
    def hash_list(self, l):
        new_l = []
        for item in l:
            if type(item) == dict:
                new_l.append(self.hash_dict(item))
            elif type(item) == list:
                new_l.append(self.hash_list(item))
            else:
                new_l.append(hash(item))
        return hash(tuple(sorted(new_l)))

    def hash_dict(self, d):
        items = []
        for key, value in d.items():
            if type(value) == dict:
                items.append(hash((key, self.hash_dict(value))))
            elif type(value) == list:
                items.append(hash((key, self.hash_list(value))))
            else:
                items.append(hash((key, value)))
        return hash(tuple(sorted(items)))
        
    def __hash__(self):
        return self.hash_dict(self)
