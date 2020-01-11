
from typing import List, Callable, Dict
from convokit import Utterance
from collections import defaultdict

class Thread:
    def __init__(self, utts: List[Utterance]):
        self.id = utts[0].meta['top_level_comment']
        self.convo_id = utts[0].root
        self.utterances = {utt.id: utt for utt in utts}
        self.users = {utt.user.id: utt.user for utt in utts}
        self.user_to_utts = defaultdict(list)
        for utt in self.get_chronological_utterance_list():
            self.user_to_utts[utt.user.id].append(utt.id)
        self.utt_to_user = dict()
        for user_id, utt_ids in self.user_to_utts.items():
            for utt_id in utt_ids:
                self.utt_to_user[utt_id] = user_id

    def iter_utterances(self, selector: Callable[[Utterance], bool] = lambda utt: True):
        for utt in self.utterances.values():
            if selector(utt):
                yield utt

    def get_utterance(self, utt_id):
        return self.utterances.get(utt_id, None)

    def check_integrity(self, verbose=True):
        if verbose: print("Checking reply-to chain of Thread {} of Convo {}".format(self.id, self.convo_id))
        utt_reply_tos = {utt.id: utt.reply_to for utt in self.iter_utterances()}
        target_utt_ids = set(list(utt_reply_tos.values()))
        speaker_utt_ids = set(list(utt_reply_tos.keys()))
        root_utt_id = target_utt_ids - speaker_utt_ids # There should only be 1 root_utt_id: convo.id

        if len(root_utt_id) != 1:
            if verbose:
                for utt_id in root_utt_id:
                    if utt_id is not None:
                        print("ERROR: Missing utterance", utt_id)
            return False
        if list(root_utt_id)[0] != self.convo_id:
            if verbose:
                print("Root utt id is not Convo id")
            return False

        if self.id not in speaker_utt_ids:
            if verbose: print("Top level comment id is missing")
            return False

        if verbose: print("No issues found.\n")
        return True

    def _print_thread_helper(self, root: str, indent: int, reply_to_dict: Dict[str, str],
                            utt_info_func: Callable[[Utterance], str]):
        print(" "*indent + utt_info_func(self.get_utterance(root)))
        children_utt_ids = [k for k, v in reply_to_dict.items() if v == root]
        for child_utt_id in children_utt_ids:
            self._print_thread_helper(root=child_utt_id, indent=indent+4,
                                      reply_to_dict=reply_to_dict, utt_info_func=utt_info_func)

    def print_thread_structure(self, utt_info_func: Callable[[Utterance], str] = lambda utt: utt.user.id):
        if not self.check_integrity(verbose=False):
            raise ValueError("Could not print thread structure: The utterance reply-to chain is broken. "
                             "Try check_integrity() to diagnose the problem.")

        reply_to_dict = {utt.id: utt.reply_to for utt in self.iter_utterances()}

        self._print_thread_helper(root=self.id, indent=0, reply_to_dict=reply_to_dict, utt_info_func=utt_info_func)

    def get_chronological_utterance_list(self, selector: Callable[[Utterance], bool] = lambda utt: True):
        return sorted([utt for utt in self.iter_utterances(selector)], key=lambda utt: utt.timestamp)

    def _get_path_from_leaf_to_root(self, leaf_utt: Utterance, root_utt: Utterance):
        if leaf_utt == root_utt:
            return [leaf_utt]
        path = [leaf_utt]
        root_id = root_utt.id
        while leaf_utt.reply_to != root_id:
            path.append(self.get_utterance(leaf_utt.reply_to))
            leaf_utt = path[-1]
        path.append(root_utt)
        return path[::-1]

    def get_root_to_leaf_paths(self):
        if not self.check_integrity(verbose=False):
            raise ValueError("Conversation failed integrity check. "
                             "It is either missing an utterance in the reply-to chain and/or has multiple root nodes. "
                             "Run check_integrity() to diagnose issues.")

        utt_reply_tos = {utt.id: utt.reply_to for utt in self.iter_utterances()}
        target_utt_ids = set(list(utt_reply_tos.values()))
        speaker_utt_ids = set(list(utt_reply_tos.keys()))
        root_utt_id = target_utt_ids - speaker_utt_ids # There should only be 1 root_utt_id: None
        assert len(root_utt_id) == 1
        root_utt = [utt for utt in self.iter_utterances() if utt.id == self.id][0]
        leaf_utt_ids = speaker_utt_ids - target_utt_ids

        paths = [self._get_path_from_leaf_to_root(self.get_utterance(leaf_utt_id), root_utt)
                 for leaf_utt_id in leaf_utt_ids]
        return paths

    def annotate_depth(self):
        for path in self.get_root_to_leaf_paths():
            for idx, utt in enumerate(path):
                utt.add_meta("depth", idx+1)

    def identify_ABC(self):
        A = self.utterances[self.id].user.id

        if len(self.users) == 1:
            return A, None, None
        else:
            ordered_utts = self.get_chronological_utterance_list()
            B = [utt for utt in ordered_utts if utt.user.id != A][0].user.id
            if len(self.users) == 2:
                return A, B, None
            else:
                C = [utt for utt in ordered_utts if utt.user.id not in {A, B}][0].user.id
                return A, B, C

    def get_user(self, user_id):
        return self.users.get(user_id, None)

    def get_AB_density(self, userA, userB):
        userA_utts = self.user_to_utts[userA]
        userB_utts = self.user_to_utts[userB]
        return len([utt_id for utt_id in userA_utts + userB_utts])

    def get_AB_density_beforeC(self, userA, userB, userC):
        userA_utts = [utt_id for utt_id in self.user_to_utts[userA]
                      if utt_id == self.id or self.get_utterance(self.get_utterance(utt_id).reply_to).user.id == userB]
        userB_utts = [utt_id for utt_id in self.user_to_utts[userB]
                      if self.get_utterance(self.get_utterance(utt_id).reply_to).user.id == userA]
        userC_utts = self.user_to_utts[userC]

        C_earliest = self.get_utterance(userC_utts[0]).timestamp
        return len([utt_id for utt_id in userA_utts + userB_utts if self.get_utterance(utt_id).timestamp < C_earliest])

    def get_C_entrance(self, userA, userB, userC):
        """

        :param userC:
        :return: Depth of entrance, 'A' or 'B' for responding to user A or user B
        """
        userC_utts = self.user_to_utts[userC]
        utt = self.get_utterance(userC_utts[0])
        target_user = self.get_utterance(utt.reply_to).user.id

        target = None
        if target_user == userA:
            target = 'A'
        elif target_user == userB:
            target = 'B'
        else:
            raise ValueError("Impossible value")
        return utt.meta['depth'], target

