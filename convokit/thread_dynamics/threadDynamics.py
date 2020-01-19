from convokit import Transformer, Corpus


class ThreadDynamics(Transformer):
    def __init__(self, prefix_len: int = 10):
        self.prefix_len = prefix_len
    """
    Annotates Conversations with the following ThreadDynamics features
    For the first n (prefix_len) utterances in the Conversation
    1. Number of unique users
    2. Depth of B's entrance
    3. Depth of C's entrance
    4. Chrono order of B's entrance
    5. Chrono order of C's entrance
    6. Does C target A? (True = 1, False = 0)
    7. Number of A->B utts (before C's entrance)
    8. Number of A->B utts (after C's entrance)
    9. Number of B->A utts (before C's entrance)
    10. Number of B->A utts (after C's entrance)
    """
    def transform(self, corpus: Corpus) -> Corpus:
        for convo in corpus.iter_conversations():
            utts_prefix = convo.get_chronological_utterance_list()[:self.prefix_len]
            retval = dict()
            retval['unique_users'] = len(set(utt.user.id for utt in utts_prefix))

            A = utts_prefix[0].user.id

            if retval['unique_users'] == 1:
                retval['B_depth'] = -1
                retval['C_depth'] = -1
                retval['B_chrono'] = -1
                retval['C_chrono'] = -1
                retval['C_targets_A'] = -1
                retval['A2B_beforeC'] = -1
                retval['A2B_afterC'] = -1
                retval['B2A_beforeC'] = -1
                retval['B2A_afterC'] = -1

            else:
                B_first_utt = [utt for utt in utts_prefix if utt.user.id != A][0]
                B = B_first_utt.user.id
                retval['B_depth'] = B_first_utt.meta['depth']
                retval['B_chrono'] = B_first_utt.meta['chrono_order']

                retval['A2B_beforeC'] = 0
                retval['B2A_beforeC'] = 0

                if retval['unique_users'] == 2:
                    retval['C_depth'] = -1
                    retval['C_chrono'] = -1
                    retval['C_targets_A'] = -1
                    retval['A2B_afterC'] = -1
                    retval['B2A_afterC'] = -1

                    for idx, utt in enumerate(utts_prefix[1:]):
                        if {utt.user.id, utts_prefix[idx].user.id} == {A, B}:
                            if utt.user.id == A:
                                retval['B2A_beforeC'] += 1
                            else:
                                retval['A2B_beforeC'] += 1

                else:
                    C_first_utt = [utt for utt in utts_prefix if utt.user.id not in {A, B}][0]
                    retval['C_depth'] = C_first_utt.meta['depth']
                    retval['C_chrono'] = C_first_utt.meta['chrono_order']
                    retval['C_targets_A'] = int(convo.get_utterance(C_first_utt.reply_to).user.id == A)

                    retval['A2B_afterC'] = 0
                    retval['B2A_afterC'] = 0

                    C_idx = retval['C_chrono'] - 1

                    for idx, utt in enumerate(utts_prefix[1:C_idx]):
                        if {utt.user.id, utts_prefix[idx].user.id} == {A, B}:
                            if utt.user.id == A:
                                retval['B2A_beforeC'] += 1
                            else:
                                retval['A2B_beforeC'] += 1

                    for idx, utt in enumerate(utts_prefix[C_idx:1]):
                        if {utt.user.id, utts_prefix[idx].user.id} == {A, B}:
                            if utt.user.id == A:
                                retval['B2A_afterC'] += 1
                            else:
                                retval['B2A_afterC'] += 1

            convo.meta['thread_dynamics'] = retval
        return corpus