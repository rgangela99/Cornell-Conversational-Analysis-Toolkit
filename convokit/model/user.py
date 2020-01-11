from functools import total_ordering
from typing import Dict, List, Optional
from .corpusUtil import warn
from .corpusObject import CorpusObject

@total_ordering
class User(CorpusObject):
    """Represents a single user in a dataset.

    :param name: name of the user.
    :type name: str
    :param utts: dictionary of utterances by the user, where key is utterance id
    :param convos: dictionary of conversations started by the user, where key is conversation id
    :param meta: arbitrary dictionary of attributes associated
        with the user.
    :type meta: dict

    :ivar name: name of the user.
    :ivar meta: dictionary of attributes associated with the user.
    """

    def __init__(self, owner=None, name: str = None, utts = None, convos = None, meta: Optional[Dict] = None):
        super().__init__(obj_type="user", owner=owner, id=name, meta=meta)
        self._name = name
        self.utterances = utts if utts is not None else dict()
        self.conversations = convos if convos is not None else dict()
        # self._split_attribs = set()
        # self._update_uid()

    # def identify_by_attribs(self, attribs: Collection) -> None:
    #     """Identify a user by a list of attributes. Sets which user info
    #     attributes should distinguish users of the same name in equality tests.
    #     For example, in the Supreme Court dataset, users are labeled with the
    #     current case id. Call this method with attribs = ["case"] to count
    #     the same person across different cases as different users.
    #
    #     By default, if this function is not called, Users are identified by name only.
    #
    #     :param attribs: Collection of attribute names.
    #     :type attribs: Collection
    #     """
    #
    #     self._split_attribs = set(attribs)
    #     # self._update_uid()

    def _get_name(self): return self._name

    def _set_name(self, value: str):
        warn("This attribute will be removed in a future release. Use User.id instead.")
        self._name = value
        # self._update_uid()

    name = property(_get_name, _set_name)

    def get_utterance(self, ut_id: str): #-> Utterance:
        """
        Get the Utterance with the specified Utterance id

        :param ut_id: The id of the Utterance
        :return: An Utterance object
        """
        return self.utterances[ut_id]

    def iter_utterances(self, selector=lambda utt: True): #-> Generator[Utterance, None, None]:
        """

        :return: An iterator of the Utterances made by the User
        """
        for v in self.utterances.values():
            if selector(v):
                yield v

    def get_utterance_ids(self, selector=lambda utt: True) -> List[str]:
        """

        :return: a List of the ids of Utterances made by the User
        """
        return list([utt.id for utt in self.iter_utterances(selector)])

    def get_conversation(self, cid: str): # -> Conversation:
        """
        Get the Conversation with the specified Conversation id

        :param cid: The id of the Conversation
        :return: A Conversation object
        """
        return self.conversations[cid]

    def iter_conversations(self, selector=lambda convo: True): # -> Generator[Conversation, None, None]:
        """

        :return: An iterator of the Conversations started by the User
        """
        for v in self.conversations.values():
            if selector(v):
                yield v

    def get_conversation_ids(self, selector=lambda convo: True) -> List[str]:
        """

        :return: a List of the ids of Conversations started by the User
        """
        return [convo.id for convo in self.iter_conversations(selector)]



    # def _update_uid(self):
    #     rep = dict()
    #     rep["name"] = self._name
    #     if self._split_attribs:
    #         rep["attribs"] = {k: self._meta[k] for k in self._split_attribs
    #                           if k in self._meta}
    #     # self.meta["uid"] = "User(" + str(sorted(rep.items())) + ")"

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, User):
            return False
        try:
            return self.id == other.id
        except AttributeError:
            return self.__dict__['_name'] == other.__dict__['_name']

    def print_user_stats(self):
        """
        Helper function for printing the number of Utterances and Conversations by the User

        :return: None
        """
        print("Number of Utterances: {}".format(len(list(self.iter_utterances()))))
        print("Number of Conversations: {}".format(len(list(self.iter_conversations()))))
    # def copy(self):
    #     """
    #     :return: A duplicate of the User with the same data and metadata
    #     """
    #     return User(name=self.name, utts=self.utterances, convos=self.conversations, meta=self.meta.copy())
