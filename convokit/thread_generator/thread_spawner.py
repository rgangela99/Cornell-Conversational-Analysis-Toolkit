from convokit import Corpus, Utterance, Speaker
import string
import random


class ThreadSpawner:
    # alternate
    def __init__(self, participant_factor: int, recip_factor: (lambda idx: 0.5),
                 expansion_factor: (lambda idx: 0.5)):
        """
        TODO: Note multiple spawns from same speaker -- generate speaker pool first!

        :param participant_factor: factor determining how often new speakers join the thread (int 1 or higher)
        :param recip_factor: factor determining the amount of reciprocity in thread (between 0 and 1)
        :param expansion_factor: tendency for speaker to expand at the top level comment (between 0 and 1)
        """
        self.participant_factor = participant_factor # starts at 1
        self.recip_factor = recip_factor
        self.expansion_factor = expansion_factor

    @staticmethod
    def construct_utt(thread_idx, comment_idx, speaker: Speaker, reply_idx, meta):
        utt = Utterance(id='{}-{}'.format(thread_idx, comment_idx),
                         root='{}-1'.format(thread_idx),
                         speaker=speaker,
                         reply_to='{}-{}'.format(thread_idx, reply_idx) if reply_idx is not None else None,
                         timestamp=comment_idx)
        utt.meta.update(meta)
        return utt

    def spawn_utt(self, utts, speaker_pool, thread_idx, comment_idx):
        # first calculate probabilities for metadata entry
        recip_prob = self.recip_factor(comment_idx)

        # probability of broaden = prob(expansion) * prob(not_root_speaker) * prob(not responded before)
        # probability of deepen = 1 - probability of broaden
        prob_root_speaker_chosen = 1.0 / (2 * self.participant_factor + 1 - 1)
        num_utts_directed_at_root = len([utt for utt in utts if utt.reply_to == utts[0].id])
        # num_utts_directed_at_root == num_spkrs_directed_at_root
        prob_responded_before = \
            num_utts_directed_at_root / (2 * self.participant_factor + 1 - 2)
        prob_broaden = (1-recip_prob) * (self.expansion_factor(comment_idx) *
                       (1 - prob_root_speaker_chosen) * (1 - prob_responded_before))
        prob_deepen = (1 - recip_prob) * (1 - prob_broaden)

        # print('reciprocate: {}, broaden: {}, deepen: {}'.format(recip_prob,
        #                                                         (1-recip_prob)*prob_broaden,
        #                                                         (1-recip_prob)*prob_deepen))
        try:
            assert 0 <= prob_deepen <= 1
            assert 0 <= prob_broaden <= 1
        except AssertionError:
            print("Prob (deepen): {}, Prob(broaden): {}, Prob(root_speaker_chosen): {}, "
                  "Prob(responded_before): {}".format(prob_deepen, prob_broaden, prob_root_speaker_chosen,
                                                      prob_responded_before))

        if random.random() <= self.recip_factor(comment_idx):
            # respond!
            prev_utt = utts[-1]
            pprev_utt = utts[-2]
            return ThreadSpawner.construct_utt(thread_idx, comment_idx, pprev_utt.speaker, prev_utt.id.split("-")[1],
                                               {'prob': recip_prob})
        else:
            # pick random speaker to respond, where random_speaker != last_speaker
            prev_speaker = utts[-1].speaker
            next_speaker = random.choice(speaker_pool)
            while next_speaker.id == prev_speaker.id:
                next_speaker = random.choice(speaker_pool)

            has_not_responded_before = lambda next_speaker, target_utt, utts: \
                all([not (utt.reply_to == target_utt and utt.speaker.id == next_speaker.id) for utt in utts])

            # pick between deepen / broaden behavior
            if random.random() < self.expansion_factor(comment_idx) and next_speaker.id != hex(1) and \
                    has_not_responded_before(next_speaker, utts[0].id, utts):
                # avoid root speaker replying to himself, avoid multiple responses to same comment
                # broaden
                return ThreadSpawner.construct_utt(thread_idx, comment_idx, next_speaker, 1, {'prob': prob_broaden})
            else:
                # pick utt to respond to that is not the root or an utt made by the speaker
                valid_choice = False
                while not valid_choice:
                    target_utt = random.choice(utts)
                    valid_choice = target_utt.reply_to is not None and target_utt.speaker.id != next_speaker.id and \
                                   has_not_responded_before(next_speaker, target_utt.id, utts)
                return ThreadSpawner.construct_utt(thread_idx, comment_idx, next_speaker,
                                                   target_utt.id.split("-")[1], {'prob': prob_deepen})

    def spawn_thread(self, thread_index: int, length: int = 20):
        assert length >= 2
        speaker_pool = [Speaker(id=hex(x)) for x in range(1, 2*self.participant_factor+1)]

        utts = [self.construct_utt(thread_index, 1, speaker_pool[0], None, {'prob': 1.0}),
                self.construct_utt(thread_index, 2, speaker_pool[1], 1, {'prob': 1.0})
                ]


        for comment_idx in range(3, length + 1):
            new_utt = self.spawn_utt(utts, speaker_pool, thread_index, comment_idx)
            utts.append(new_utt)

        return utts







