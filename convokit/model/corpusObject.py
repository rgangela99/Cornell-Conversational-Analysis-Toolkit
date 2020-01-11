from warnings import warn
# from abc import ABC, abstractmethod
from .convoKitMeta import ConvoKitMeta


class CorpusObject:

    def __init__(self, obj_type: str, owner=None, id=None, meta=None):
        self.obj_type = obj_type  # utterance, user, conversation
        self._owner = owner
        if meta is None:
            meta = dict()
        self.meta = self.init_meta(meta)
        self._id = id

    def get_owner(self):
        return self._owner

    def set_owner(self, owner):
        self._owner = owner
        if owner is not None:
            self.meta = self.init_meta(self.meta)

    owner = property(get_owner, set_owner)

    def init_meta(self, meta):
        if self._owner is None:
            return meta
        else:
            ck_meta = ConvoKitMeta(self.owner.meta_index, self.obj_type)
            for key, value in meta.items():
                ck_meta[key] = value
            return ck_meta

    def get_id(self):
        return self._id

    def set_id(self, value):
        if not isinstance(value, str) and value is not None:
            self._id = str(value)
            warn("{} id must be a string. Input has been casted to string.".format(self.obj_type))
        self._id = value

    id = property(get_id, set_id)

    # def __eq__(self, other):
    #     if type(self) != type(other): return False
    #     # do not compare 'utterances' and 'conversations' in User.__dict__. recursion loop will occur.
    #     self_keys = set(self.__dict__).difference(['_owner', 'meta', 'utterances', 'conversations'])
    #     other_keys = set(other.__dict__).difference(['_owner', 'meta', 'utterances', 'conversations'])
    #     return self_keys == other_keys and all([self.__dict__[k] == other.__dict__[k] for k in self_keys])

    def retrieve_meta(self, key: str):
        return self.meta[key]

    def add_meta(self, key: str, value) -> None:
        """
        Adds a key-value pair to the metadata of the User

        :return: None
        """
        self.meta[key] = value

    def get_info(self, key):
        """
            Gets attribute <key> of the user. Returns None if the user does not have this attribute.

            :param key: name of attribute
            :return: attribute <key>
        """

        return self.meta.get(key, None)

    def set_info(self, key, value):
        """
            Sets attribute <key> of the user to <value>.

            :param key: name of attribute
            :param value: value to set
            :return: None
        """

        self.meta[key] = value

    def del_info(self, key):
        if key in self.meta:
            del self.meta[key]

    def __str__(self):
        return "{}('id': {}, 'meta': {})".format(self.obj_type.capitalize(),
                                                 self.id,
                                                 self.meta)

    def __hash__(self):
        return hash(self.obj_type + self.id)

    def __repr__(self):
        copy = self.__dict__.copy()
        if 'utterances' in copy:
            del copy['utterances']
        if 'conversations' in copy:
            del copy['conversations']
        try:
            return self.obj_type.capitalize() + "(" + str(copy) + ")"
        except AttributeError: # for backwards compatibility when corpus objects are saved as binary data, e.g. wikiconv
            return "(" + str(copy) + ")"
