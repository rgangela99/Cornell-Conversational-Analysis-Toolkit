from convokit import Corpus, Utterance, User
import string
import random


class ThreadSpawner:
    # alternate
    def __init__(self, participant_factor: int, recip_factor: (lambda idx: 0.5),
                 expansion_factor: (lambda idx: 0.5)):
        """
        TODO: Note multiple spawns from same user -- generate user pool first!

        :param participant_factor: factor determining how often new users join the thread (int 1 or higher)
        :param recip_factor: factor determining the amount of reciprocity in thread (between 0 and 1)
        :param expansion_factor: tendency for user to expand at the top level comment (between 0 and 1)
        """
        self.participant_factor = participant_factor # starts at 1
        self.recip_factor = recip_factor
        self.expansion_factor = expansion_factor

    @staticmethod
    def construct_utt(thread_idx, comment_idx, user: User, reply_idx):
        return Utterance(id='{}-{}'.format(thread_idx, comment_idx),
                         root='{}-1'.format(thread_idx),
                         user=user,
                         reply_to='{}-{}'.format(thread_idx, reply_idx) if reply_idx is not None else None,
                         timestamp=comment_idx)

    def spawn_utt(self, utts, user_pool, thread_idx, comment_idx):
        if random.random() < self.recip_factor(comment_idx):
            # respond!
            prev_utt = utts[-1]
            pprev_utt = utts[-2]
            return ThreadSpawner.construct_utt(thread_idx, comment_idx,
                                               pprev_utt.user,
                                               prev_utt.id.split("-")[1])
        else:
            # pick random user to respond, where random_user != last_user
            prev_user = utts[-1].user
            next_user = random.choice(user_pool)
            while next_user.id == prev_user.id:
                next_user = random.choice(user_pool)

            has_not_responded_before = lambda next_user, target_utt, utts: all([not (utt.reply_to == target_utt and utt.user.id == next_user.id) for utt in utts])
            # pick between deepen / broaden behavior
            if random.random() < self.expansion_factor(comment_idx) and next_user.id != hex(1) and \
                    has_not_responded_before(next_user, utts[0].id, utts): # avoid root user replying to himself, avoid multiple responses to same comment
                # broaden
                return ThreadSpawner.construct_utt(thread_idx, comment_idx, next_user, 1)
            else:
                # pick utt to respond to that is not the root or an utt made by the user
                valid_choice = False
                while not valid_choice:
                    target_utt = random.choice(utts)
                    valid_choice = target_utt.reply_to is not None and target_utt.user.id != next_user.id and has_not_responded_before(next_user, target_utt.id, utts)
                return ThreadSpawner.construct_utt(thread_idx, comment_idx, next_user, target_utt.id.split("-")[1])

    def spawn_thread(self, thread_index: int, length: int = 20):
        assert length >= 2
        user_pool = [User(id=hex(x)) for x in range(1, 2*self.participant_factor+1)]

        utts = [self.construct_utt(thread_index, 1, user_pool[0], None),
                self.construct_utt(thread_index, 2, user_pool[1], 1)
                ]


        for comment_idx in range(3, length + 1):
            new_utt = self.spawn_utt(utts, user_pool, thread_index, comment_idx)
            utts.append(new_utt)

        return utts







