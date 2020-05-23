from collections import MutableMapping
from convokit.util import warn
from .convoKitIndex import ConvoKitIndex
import json


class ConvoKitMeta(MutableMapping, dict):
    """
    See reference: https://stackoverflow.com/questions/7760916/correct-usage-of-a-getter-setter-for-dictionary-values
    """
    def __init__(self, convokit_index, obj_type):
        self.index: ConvoKitIndex = convokit_index
        self.obj_type = obj_type

    def __getitem__(self, item):
        return dict.__getitem__(self, item)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            warn("Metadata keys must be strings. Input key has been casted to a string.")
        key = str(key)
        if key not in self.index.indices[self.obj_type]:
            # update Corpus index
            try:
                json.dumps(value)
                self.index.update_index(self.obj_type, key=key, class_type=str(type(value)))
            except (TypeError, OverflowError):   # unserializable
                self.index.update_index(self.obj_type, key=key, class_type="bin")
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self.index.del_from_index(self.obj_type, key)

    def __iter__(self):
        return dict.__iter__(self)

    def __len__(self):
        return dict.__len__(self)

    def __contains__(self, x):
        return dict.__contains__(self, x)

    def to_dict(self):
        return self.__dict__
