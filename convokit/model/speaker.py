from functools import total_ordering
from typing import Dict, List, Optional
from convokit.util import deprecation
from .corpusObject import CorpusObject

@total_ordering
class Speaker(CorpusObject):
    """
    Represents a single speaker in a dataset.

    :param id: id of the speaker.
    :type id: str
    :param utts: dictionary of utterances by the speaker, where key is utterance id
    :param convos: dictionary of conversations started by the speaker, where key is conversation id
    :param meta: arbitrary dictionary of attributes associated
        with the speaker.
    :type meta: dict

    :ivar id: id of the speaker.
    :ivar meta: A dictionary-like view object providing read-write access to
        speaker-level metadata.
    """

    def __init__(self, owner=None, id: str = None, name: str = None, utts=None, convos=None, meta: Optional[Dict] = None):
        name_var = id if id is not None else name # to be deprecated
        super().__init__(obj_type="speaker", owner=owner, id=name_var, meta=meta)
        self.utterances = utts if utts is not None else dict()
        self.conversations = convos if convos is not None else dict()
        # self._split_attribs = set()
        # self._update_uid()

    # def identify_by_attribs(self, attribs: Collection) -> None:
    #     """Identify a speaker by a list of attributes. Sets which speaker info
    #     attributes should distinguish speakers of the same name in equality tests.
    #     For example, in the Supreme Court dataset, speakers are labeled with the
    #     current case id. Call this method with attribs = ["case"] to count
    #     the same person across different cases as different speakers.
    #
    #     By default, if this function is not called, speakers are identified by name only.
    #
    #     :param attribs: Collection of attribute names.
    #     :type attribs: Collection
    #     """
    #
    #     self._split_attribs = set(attribs)
    #     # self._update_uid()

    def _get_name(self):
        deprecation("speaker.name", "speaker.id")
        return self._id

    def _set_name(self, value: str):
        deprecation("speaker.name", "speaker.id")
        self._id = value
        # self._update_uid()

    name = property(_get_name, _set_name)

    def _add_utterance(self, utt):
        self.utterances[utt.id] = utt

    def _add_conversation(self, convo):
        self.conversations[convo.id] = convo

    def get_utterance(self, ut_id: str): #-> Utterance:
        """
        Get the Utterance with the specified Utterance id

        :param ut_id: The id of the Utterance
        :return: An Utterance object
        """
        return self.utterances[ut_id]

    def iter_utterances(self, selector=lambda utt: True): #-> Generator[Utterance, None, None]:
        """

        :return: An iterator of the Utterances made by the speaker
        """
        for v in self.utterances.values():
            if selector(v):
                yield v

    def get_utterance_ids(self, selector=lambda utt: True) -> List[str]:
        """

        :return: a List of the ids of Utterances made by the speaker
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

        :return: An iterator of the Conversations started by the speaker
        """
        for v in self.conversations.values():
            if selector(v):
                yield v

    def get_conversation_ids(self, selector=lambda convo: True) -> List[str]:
        """

        :return: a List of the ids of Conversations started by the speaker
        """
        return [convo.id for convo in self.iter_conversations(selector)]



    # def _update_uid(self):
    #     rep = dict()
    #     rep["name"] = self._name
    #     if self._split_attribs:
    #         rep["attribs"] = {k: self._meta[k] for k in self._split_attribs
    #                           if k in self._meta}
    #     # self.meta["uid"] = "speaker(" + str(sorted(rep.items())) + ")"

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Speaker):
            return False
        try:
            return self.id == other.id
        except AttributeError:
            return self.__dict__['_name'] == other.__dict__['_name']

    def print_speaker_stats(self):
        """
        Helper function for printing the number of Utterances and Conversations by the speaker

        :return: None
        """
        print("Number of Utterances: {}".format(len(list(self.iter_utterances()))))
        print("Number of Conversations: {}".format(len(list(self.iter_conversations()))))
    # def copy(self):
    #     """
    #     :return: A duplicate of the speaker with the same data and metadata
    #     """
    #     return speaker(name=self.name, utts=self.utterances, convos=self.conversations, meta=self.meta.copy())
