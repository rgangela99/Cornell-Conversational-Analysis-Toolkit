from typing import Dict, List, Callable, Generator, Optional, Set
from .utterance import Utterance
from .user import User
from .corpusUtil import warn
from .corpusObject import CorpusObject
from collections import defaultdict
from .utteranceNode import UtteranceNode

class Conversation(CorpusObject):
    """Represents a discrete subset of utterances in the dataset, connected by a
    reply-to chain.

    :param owner: The Corpus that this Conversation belongs to
    :param cid: The unique ID of this Conversation
    :param utterances: A list of the IDs of the Utterances in this Conversation
    :param meta: Table of initial values for conversation-level metadata

    :ivar id: the ID of the Conversation
    :ivar meta: A dictionary-like view object providing read-write access to
        conversation-level metadata.
    """

    def __init__(self, owner, id: Optional[str] = None,
                 utterances: Optional[List[str]] = None,
                 meta: Optional[Dict] = None):
        super().__init__(obj_type="conversation", owner=owner, id=id, meta=meta)
        self._owner = owner
        self._utterance_ids: List[str] = utterances
        self._user_ids = None
        self.tree: Optional[UtteranceNode] = None

    def _add_utterance(self, utt: Utterance):
        self._utterance_ids.append(utt.id)
        self._user_ids = None
        self.tree = None

    def get_utterance_ids(self) -> List[str]:
        """Produces a list of the unique IDs of all utterances in the
        Conversation, which can be used in calls to get_utterance() to retrieve
        specific utterances. Provides no ordering guarantees for the list.

        :return: a list of IDs of Utterances in the Conversation
        """
        # we construct a new list instead of returning self._utterance_ids in
        # order to prevent the user from accidentally modifying the internal
        # ID list (since lists are mutable)
        return [ut_id for ut_id in self._utterance_ids]

    def get_utterance(self, ut_id: str) -> Utterance:
        """Looks up the Utterance associated with the given ID. Raises a
        KeyError if no utterance by that ID exists.

        :return: the Utterance with the given ID
        """
        # delegate to the owner Corpus since Conversation does not itself own
        # any Utterances
        return self._owner.get_utterance(ut_id)

    def iter_utterances(self, selector: Callable[[Utterance], bool] = lambda utt: True) -> Generator[Utterance, None, None]:
        """Generator allowing iteration over all utterances in the Conversation.
        Provides no ordering guarantees.

        :return: Generator that produces Users
        """
        for ut_id in self._utterance_ids:
            utt = self._owner.get_utterance(ut_id)
            if selector(utt):
                yield utt

    def get_usernames(self) -> List[str]:
        """Produces a list of names of all users in the Conversation, which can
        be used in calls to get_user() to retrieve specific users. Provides no
        ordering guarantees for the list.

        :return: a list of usernames
        """
        warn("This function is deprecated and will be removed in a future release. Use get_user_ids() instead.")
        if self._user_ids is None:
            # first call to get_usernames or iter_users; precompute cached list
            # of usernames
            self._user_ids = set()
            for ut_id in self._utterance_ids:
                ut = self._owner.get_utterance(ut_id)
                self._user_ids.add(ut.user.name)
        return list(self._user_ids)

    def get_user_ids(self) -> List[str]:
        """Produces a list of ids of all users in the Conversation, which can
        be used in calls to get_user() to retrieve specific users. Provides no
        ordering guarantees for the list.

        :return: a list of usernames
        """
        if self._user_ids is None:
            # first call to get_usernames or iter_users; precompute cached list
            # of usernames
            self._user_ids = set()
            for ut_id in self._utterance_ids:
                ut = self._owner.get_utterance(ut_id)
                self._user_ids.add(ut.user.name)
        return list(self._user_ids)

    def get_user(self, username: str) -> User:
        """
        Looks up the User with the given name. Raises a KeyError if no user
        with that name exists.

        :return: the User with the given username
        """
        # delegate to the owner Corpus since Conversation does not itself own
        # any Utterances
        return self._owner.get_user(username)

    def iter_users(self, selector: Callable[[User], bool] = lambda user: True) -> Generator[User, None, None]:
        """
        Generator allowing iteration over all users in the Conversation.
        Provides no ordering guarantees.

        :return: Generator that produces Users.
        """
        if self._user_ids is None:
            # first call to get_ids or iter_users; precompute cached list of usernames
            self._user_ids = set()
            for ut_id in self._utterance_ids:
                ut = self._owner.get_utterance(ut_id)
                self._user_ids.add(ut.user.id)
        for user_id in self._user_ids:
            yield self._owner.get_user(user_id)

    def get_chronological_user_list(self, selector: Callable[[User], bool] = lambda user: True):
        """
        Get the users in the conversation sorted in chronological order (users may appear more than once)

        :param selector: (lambda) function for which users should be included; all users are included by default
        :return: list of users for each chronological utterance
        """
        chrono_utts = sorted(list(self.iter_utterances()), key=lambda utt: utt.timestamp)
        return [utt.user for utt in chrono_utts if selector(utt.user)]

    def check_integrity(self, verbose=True):
        if verbose: print("Checking reply-to chain of Conversation", self.id)
        utt_reply_tos = {utt.id: utt.reply_to for utt in self.iter_utterances()}
        target_utt_ids = set(list(utt_reply_tos.values()))
        speaker_utt_ids = set(list(utt_reply_tos.keys()))
        root_utt_id = target_utt_ids - speaker_utt_ids # There should only be 1 root_utt_id: None

        if len(root_utt_id) != 1:
            if verbose:
                for utt_id in root_utt_id:
                    if utt_id is not None:
                        warn("ERROR: Missing utterance {}".format(utt_id))
            return False
        else:
            root_id = list(root_utt_id)[0]
            if root_id is not None:
                if verbose: warn("ERROR: Missing utterance {}".format(root_id))
                return False

        # sanity check
        utts_replying_to_none = 0
        for utt in self.iter_utterances():
            if utt.reply_to is None:
                utts_replying_to_none += 1

        if utts_replying_to_none > 1:
            if verbose: warn("ERROR: Found more than one Utterance replying to None.")
            return False

        if verbose: print("No issues found.\n")
        return True

    def initialize_tree_structure(self):
        if not self.check_integrity(verbose=False):
            raise ValueError("Conversation {} reply-to chain does not form a valid tree.".format(self.id))

        root_node_id = None
        # Find root node
        for utt in self.iter_utterances():
            if utt.reply_to is None:
                root_node_id = utt.id

        parent_to_children_ids = defaultdict(list)
        for utt in self.iter_utterances():
            parent_to_children_ids[utt.reply_to].append(utt.id)

        wrapped_utts = {utt.id: UtteranceNode(utt) for utt in self.iter_utterances()}

        for parent_id, wrapped_utt in wrapped_utts.items():
            wrapped_utt.set_children([wrapped_utts[child_id] for child_id in parent_to_children_ids[parent_id]])

        self.tree = wrapped_utts[root_node_id]

    def traverse(self, traversal_type: str, as_utterance: bool = True):
        """
        Traverse through the Conversation tree structure in a breadth-first search ('bfs'), depth-first search (dfs),
        pre-order ('preorder'), or post-order ('postorder') way.

        :param traversal_type: dfs, bfs, preorder, or postorder
        :param as_utterance: whether the iterator should yield the utterance (True) or the utterance node (False)
        :return: an iterator of the utterances or utterance nodes
        """
        if self.tree is None:
            self.initialize_tree_structure()
            if self.tree is None:
                raise ValueError("Failed to traverse because Conversation reply-to chain does not form a valid tree.")

        traversals = {'bfs': self.tree.bfs_traversal,
                      'dfs': self.tree.dfs_traversal,
                      'preorder': self.tree.pre_order,
                      'postorder': self.tree.post_order}

        for utt_node in traversals[traversal_type]():
            yield utt_node.utt if as_utterance else utt_node

    def get_subtree(self, root_utt_id):
        """
        Get the utterance node of the specified input id

        :param root_utt_id: id of the root node that the subtree starts from
        :return: UtteranceNode object
        """
        if self.tree is None:
            self.initialize_tree_structure()
            if self.tree is None:
                raise ValueError("Failed to traverse because Conversation reply-to chain does not form a valid tree.")

        for utt_node in self.tree.bfs_traversal():
            if utt_node.utt.id == root_utt_id:
                return utt_node

    def _print_convo_helper(self, root: str, indent: int, reply_to_dict: Dict[str, str],
                            utt_info_func: Callable[[Utterance], str],
                            limit=None) -> None:
        """
        Helper function for print_conversation_structure()
        """
        if limit is not None:
            if self.get_utterance(root).meta['order'] > limit:
                return
        print(" "*indent + utt_info_func(self.get_utterance(root)))
        children_utt_ids = [k for k, v in reply_to_dict.items() if v == root]
        for child_utt_id in children_utt_ids:
            self._print_convo_helper(root=child_utt_id, indent=indent+4,
                                     reply_to_dict=reply_to_dict, utt_info_func=utt_info_func, limit=limit)

    def print_conversation_structure(self, utt_info_func: Callable[[Utterance], str] = lambda utt: utt.user.id, limit: int = None) -> None:
        """
        Prints an indented representation of utterances in the Conversation with conversation reply-to structure determining the indented level. The details of each utterance to be printed can be configured.

        If limit is set to a value other than None, this will annotate utterances with an 'order' metadata indicating their temporal order in the conversation, where the first utterance in the conversation is annotated with 1.

        :param utt_info_func: callable function taking an utterance as input and returning a string of the desired
                              utterance information. By default, this is a lambda function returning the utterance's user's id
        :param limit: maximum number of utterances to print out. if k, this includes the first k utterances.
        :return: None. Prints to stdout.
        """
        if not self.check_integrity(verbose=False):
            raise ValueError("Could not print conversation structure: The utterance reply-to chain is broken. "
                             "Try check_integrity() to diagnose the problem.")

        if limit is not None:
            assert isinstance(limit, int)
            for idx, utt in enumerate(self.get_chronological_utterance_list()):
                utt.meta['order'] = idx + 1

        root_utt_id = [utt for utt in self.iter_utterances() if utt.reply_to is None][0].id
        reply_to_dict = {utt.id: utt.reply_to for utt in self.iter_utterances()}

        self._print_convo_helper(root=root_utt_id, indent=0, reply_to_dict=reply_to_dict,
                                 utt_info_func=utt_info_func, limit=limit)

    def get_chronological_utterance_list(self, selector: Callable[[Utterance], bool] = lambda utt: True):
        """
        Get the utterances in the conversation sorted in increasing order of timestamp

        :param selector: function for which utterances should be included; all utterances are included by default
        :return: list of utterances, sorted by timestamp
        """
        return sorted([utt for utt in self.iter_utterances(selector)], key=lambda utt: utt.timestamp)

    def _get_path_from_leaf_to_root(self, leaf_utt: Utterance, root_utt: Utterance) -> List[Utterance]:
        """
        Helper function for get_root_to_leaf_paths, which returns the path for a given leaf_utt and root_utt
        """
        if leaf_utt == root_utt:
            return [leaf_utt]
        path = [leaf_utt]
        root_id = root_utt.id
        while leaf_utt.reply_to != root_id:
            path.append(self.get_utterance(leaf_utt.reply_to))
            leaf_utt = path[-1]
        path.append(root_utt)
        return path[::-1]

    def get_root_to_leaf_paths(self) -> List[List[Utterance]]:
        """
        Get the paths (stored as a list of lists of utterances) from the root to each of the leaves
        in the conversational tree

        :return: List of lists of Utterances
        """
        if not self.check_integrity(verbose=False):
            raise ValueError("Conversation failed integrity check. "
                             "It is either missing an utterance in the reply-to chain and/or has multiple root nodes. "
                             "Run check_integrity() to diagnose issues.")

        utt_reply_tos = {utt.id: utt.reply_to for utt in self.iter_utterances()}
        target_utt_ids = set(list(utt_reply_tos.values()))
        speaker_utt_ids = set(list(utt_reply_tos.keys()))
        root_utt_id = target_utt_ids - speaker_utt_ids # There should only be 1 root_utt_id: None
        assert len(root_utt_id) == 1
        root_utt = [utt for utt in self.iter_utterances() if utt.reply_to is None][0]
        leaf_utt_ids = speaker_utt_ids - target_utt_ids

        paths = [self._get_path_from_leaf_to_root(self.get_utterance(leaf_utt_id), root_utt)
                 for leaf_utt_id in leaf_utt_ids]
        return paths

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Conversation):
            return False
        return self.id == other.id and set(self._utterance_ids) == set(other._utterance_ids)

    def __str__(self):
        return "Conversation('id': {}, 'utterances': {}, 'meta': {})".format(repr(self.id), self._utterance_ids, self.meta)
