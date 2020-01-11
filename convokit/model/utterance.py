from typing import Dict, List, Collection, Callable, Set, Generator, Tuple, Optional, ValuesView
from .user import User
from .corpusObject import CorpusObject


class Utterance(CorpusObject):
    """Represents a single utterance in the dataset.

    :param id: the unique id of the utterance.
    :param user: the user giving the utterance.
    :param root: the id of the root utterance of the conversation.
    :param reply_to: id of the utterance this was a reply to.
    :param timestamp: timestamp of the utterance. Can be any
        comparable type.
    :param text: text of the utterance.
    :type text: str

    :ivar id: the unique id of the utterance.
    :ivar user: the user giving the utterance.
    :ivar root: the id of the root utterance of the conversation.
    :ivar reply_to: id of the utterance this was a reply to.
    :ivar timestamp: timestamp of the utterance.
    :ivar text: text of the utterance.
    """

    def __init__(self, owner=None, id: Optional[str] = None, user: Optional[User] = None,
                 root: Optional[str] = None, reply_to: Optional[str] = None,
                 timestamp: Optional[int] = None, text: Optional[str] = None,
                 meta: Optional[Dict] = None):
        super().__init__(obj_type="utterance", owner=owner, id=id, meta=meta)
        self.user = user
        self.root = root
        self.reply_to = reply_to
        self.timestamp = timestamp # int(timestamp) if timestamp is not None else timestamp
        self.text = text

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Utterance):
            return False
        try:
            return self.id == other.id and self.root == other.root and self.reply_to == other.reply_to and \
                   self.user == other.user and self.timestamp == other.timestamp and self.text == other.text
        except AttributeError: # for backwards compatibility with wikiconv
            return self.__dict__ == other.__dict__

    def __str__(self):
        return "Utterance('id': {}, 'root': {}, 'reply-to': {}, " \
               "'user': {}, 'timestamp': {}, 'text': {}, 'meta': {})".format(repr(self.id),
                                                                             self.root,
                                                                             self.reply_to,
                                                                             self.user,
                                                                             self.timestamp,
                                                                             repr(self.text),
                                                                             self.meta)


