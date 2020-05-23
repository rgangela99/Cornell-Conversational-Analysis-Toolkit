import pkg_resources
from convokit.model import Corpus, Speaker, Utterance
from collections import defaultdict
from typing import Callable, Tuple, List, Dict, Optional, Collection, Union
from .coordinationScore import CoordinationScore, CoordinationWordCategories

from convokit.transformer import Transformer

class Coordination(Transformer):
    """Encapsulates computation of coordination-based features for a particular
    corpus.
    
    Note: labeling method is slightly different from that used in the paper --
    we no longer match words occurring in the middle of other words and that
    immediately follow an apostrophe. Most notably, we no longer separately
    count the "all" in "y'all."
    """

    def __init__(self, **thresh):
        self.thresh = thresh
        self.corpus = None
        self.precomputed = False

    def fit(self, corpus: Corpus, y=None):
        """Learn coordination information for the given corpus."""
        self.corpus = corpus
        self.precompute(corpus)

    def transform(self, corpus: Corpus) -> Corpus:
        """Generate coordination scores for the corpus you called fit on."""
        if corpus != self.corpus:
            raise Exception("Coordination: must fit and transform on same corpus")
        if not self.precomputed:
            raise Exception("Must fit before calling transform")

        pair_scores = self.pairwise_scores(corpus, corpus.speaking_pairs(), **self.thresh)
        for (speaker, target), score in pair_scores.items():
            if "coord-score" not in speaker.meta:
                speaker.meta["coord-score"] = {}
            speaker.meta["coord-score"][target.id] = score

            assert isinstance(speaker, Speaker)


        return corpus

    def precompute(self, corpus) -> None:
        """Call this to run the time-consuming annotation process explicitly.
        For example, this lets you save the annotated coordination object as a
        pickle to cache the precomputation results."""

        if not self.precomputed:
            self._compute_liwc_reverse_dict()
            self._annot_liwc_cats(corpus)
            self.precomputed = True

    def score(self, corpus: Corpus, speakers: Collection[Union[Speaker, str]],
              group: Collection[Union[Speaker, str]], focus: str = "speakers",
              speaker_thresh: int = 0, target_thresh: int = 3,
              utterances_thresh: int = 0, speaker_thresh_indiv: int = 0,
              target_thresh_indiv: int = 0,
              utterances_thresh_indiv: int = 0,
              utterance_thresh_func: Optional[Callable[[Tuple[Utterance, Utterance]], bool]] = None,
              split_by_attribs: Optional[List[str]] = None,
              speaker_attribs: Optional[Dict] = None, target_attribs: Optional[Dict] = None) -> CoordinationScore:
        """Computes the coordination scores for each speaker, given a set of speakers and a group of targets.

        :param corpus: Corpus to compute scores on
        :param speakers: A collection of speaker ids or speaker objects corresponding
            to the speakers we want to compute scores for.
        :param group: A collection of speaker ids or speaker objects corresponding to
            the group of targets.
        :param focus: Either "speakers" or "targets". If "speakers", treat the
            set of targets for a particular speaker as a single person (i.e.
            concatenate all of their utterances); the returned dictionary will
            have speakers as keys. If "targets", treat the set of
            speakers for a particular target as a single person; the returned
            dictionary will have targets as keys.
        :param speaker_thresh: Thresholds based on minimum number of times the speaker uses each coordination marker.
        :param target_thresh: Thresholds based on minimum number of times the target uses each coordination marker.
        :param utterances_thresh: Thresholds based on the minimum number of utterances for each speaker.
        :param speaker_thresh_indiv: Like `speaker_thresh` but only considers the utterances between a speaker and a single target; thresholds whether the utterances for a single target should be considered for a particular speaker.
        :param target_thresh_indiv: Like `target_thresh` but thresholds whether a single target's utterances should be considered for a particular speaker.
        :param utterances_thresh_indiv: Like `utterances_thresh` but thresholds whether a single target's utterances should be considered for a particular speaker.
        :param utterance_thresh_func: Optional utterance-level threshold function that takes in a speaker `Utterance` and the `Utterance` the speaker replied to, and returns a `bool` corresponding to whether or not to include the utterance in scoring.
        :param split_by_attribs: Utterance meta attributes to split speakers by when tallying coordination (e.g. in supreme court transcripts, you may want to treat the same lawyer as a different person across different cases --- see coordination examples)
        :param speaker_attribs: attribute names and values the speaker must have
        :param target_attribs: attribute names and values the target must have

        :return: A :class:`CoordinationScore` object corresponding to the coordination scores for each speaker.
        """
        if corpus != self.corpus:
            raise Exception("Coordination: must fit and score on same corpus")
        if not self.precomputed:
            raise Exception("Must fit before calling score")

        if split_by_attribs is None: split_by_attribs = []
        if speaker_attribs is None: speaker_attribs = dict()
        if target_attribs is None: target_attribs = dict()

        #self.precompute()
        speakers = set(speakers)
        group = set(group)

        utterances = []
        for utt in corpus.iter_utterances():
            speaker = utt.speaker
            if speaker in speakers:
                if utt.reply_to is not None:
                    reply_to = corpus.get_utterance(utt.reply_to)
                    target = reply_to.speaker
                    if target in group:
                        utterances.append(utt)
        return self.scores_over_utterances(corpus, speakers, utterances,
            speaker_thresh, target_thresh, utterances_thresh,
            speaker_thresh_indiv, target_thresh_indiv,
            utterances_thresh_indiv, utterance_thresh_func,
            focus, split_by_attribs, speaker_attribs, target_attribs)

    def pairwise_scores(self, corpus: Corpus,
                        pairs: Collection[Tuple[Union[Speaker, str], Union[Speaker, str]]],
                        speaker_thresh: int = 0, target_thresh: int = 3,
                        utterances_thresh: int = 0, speaker_thresh_indiv: int = 0,
                        target_thresh_indiv: int = 0, utterances_thresh_indiv: int = 0,
                        utterance_thresh_func: Optional[Callable[[Tuple[Utterance, Utterance]], bool]] = None)\
            -> CoordinationScore:
        """Computes all pairwise coordination scores given a collection of
        (speaker, target) pairs.
        
        :param corpus: Corpus to compute scores on
        :param pairs: collection of (speaker id, target id) pairs
        :type pairs: Collection
        
        Also accepted: all threshold arguments accepted by :func:`score()`.

        :return: A :class:`CoordinationScore` object corresponding to the
            coordination scores for each (speaker, target) pair.
        """
        if corpus != self.corpus:
            raise Exception("Coordination: must fit and score on same corpus")
        if not self.precomputed:
            raise Exception("Must fit before calling score")

        pairs = set(pairs)
        any_speaker = next(iter(pairs))[0]
        if isinstance(any_speaker, str):
            pairs_utts = corpus.pairwise_exchanges(lambda x, y:
                (x.name, y.name) in pairs, speaker_names_only=True)
        else:
            pairs_utts = corpus.pairwise_exchanges(lambda x, y:
                (x, y) in pairs, speaker_names_only=False)
        all_scores = CoordinationScore()
        for (speaker, target), utterances in pairs_utts.items():
            scores = self.scores_over_utterances(corpus, [speaker], utterances, speaker_thresh, target_thresh,
                                                 utterances_thresh, speaker_thresh_indiv, target_thresh_indiv,
                                                 utterances_thresh_indiv, utterance_thresh_func)
            if len(scores) > 0: # scores.values() will be length 0 or 1
                all_scores[speaker, target] = list(scores.values())[0]
        return all_scores

    def score_report(self, corpus: Corpus, scores: CoordinationScore):
        """Create a "score report" of aggregate scores given a score output
        produced by `score` or `pairwise_scores`.

        - aggregate 1: average scores only over speakers with a score for each
            coordination marker.
        - aggregate 2: fill in missing scores for a speaker by using the group
            score for each missing marker. (assumes different people in a group
            coordinate the same way.)
        - aggregate 3: fill in missing scores for a speaker by using the average
            score over the markers we can compute coordination for for that 
            speaker. (assumes a speaker coordinates the same way across different
            coordination markers.)

        :param corpus: Corpus to compute scores on
        :param scores: Scores to produce a report for.
        :type scores: CoordinationScore

        :return: A tuple (marker_a1, marker, agg1, agg2, agg3):

            - marker_a1 is a dictionary of aggregate scores by marker,
                using the scores only over speakers included in Aggregate 1.
            - marker is a dictionary of aggregate scores by marker,
                using the scores of all speakers with a coordination score for
                that marker.
            - agg1, agg2 and agg3 are Aggregate 1, 2 and 3 scores respectively.

        """
        if corpus != self.corpus:
            raise Exception("Coordination: must fit and score on same corpus")
        if not self.precomputed:
            raise Exception("Must fit before calling score")

        marker_a1 = scores.averages_by_marker(strict_thresh=True)  
        marker = scores.averages_by_marker()
        agg1 = scores.aggregate(method=1)
        agg2 = scores.aggregate(method=2)
        agg3 = scores.aggregate(method=3)
        return marker_a1, marker, agg1, agg2, agg3

    # helper functions
    def _compute_liwc_reverse_dict(self) -> None:
        with open(pkg_resources.resource_filename("convokit",
            "data/coord-liwc-patterns.txt"), "r") as f:
            all_words = []
            for line in f:
                cat, pat = line.strip().split("\t")
                #if cat == "auxverb": print(cat, pat)
                # use "#" to mark word boundary
                words = pat.replace("\\b", "#").split("|")
                all_words += [(w[1:], cat) for w in words]
            self.liwc_trie = Coordination.make_trie(all_words)

    @staticmethod
    def make_trie(words) -> Dict:
        root = {}
        for word, cat in words:
            cur = root
            for c in word:
                cur = cur.setdefault(c, {})
            if "$" not in cur:   # use "$" as end-of-word symbol
                cur["$"] = {cat}
            else:
                cur["$"].add(cat)
        return root

    def _annot_liwc_cats(self, corpus) -> None:
        # add liwc_categories field to each utterance
        word_chars = set("abcdefghijklmnopqrstuvwxyz0123456789_")
        for utt in corpus.iter_utterances():
            cats = set()
            last = None
            cur = None
            text = utt.text.lower() + " "
            #if "'" in text: print(text)
            for i, c in enumerate(text):
                # slightly different from regex: won't match word after an
                #   apostrophe unless the apostrophe starts the word
                #   -- avoids false positives
                if last not in word_chars and c in word_chars and (last != "'"
                    or not cur):
                    cur = self.liwc_trie
                if cur:
                    if c in cur and c != "#" and c != "$":
                        if c not in word_chars:
                            if "#" in cur and "$" in cur["#"]:
                                cats |= cur["#"]["$"]  # finished current word
                        cur = cur[c]
                    elif c not in word_chars and last in word_chars and \
                        "#" in cur:
                        cur = cur["#"]
                    else:
                        cur = None
                if cur and "$" in cur:
                    cats |= cur["$"]
                last = c
            utt.meta["liwc-categories"] = cats

    @staticmethod
    def _annot_speaker(speaker: Speaker, utt: Utterance, split_by_attribs):
        return (speaker, tuple([utt.meta[attrib] if attrib in utt.meta else None
                             for attrib in split_by_attribs]))

    @staticmethod
    def _utterance_has_attribs(utterance, desired_attribs) -> bool:
        for attrib, attrib_val in desired_attribs.items():
            if utterance.meta[attrib] != attrib_val:
                return False
        return True

    def scores_over_utterances(self, corpus: Corpus, speakers: Collection[Union[Speaker, str]], utterances,
                               speaker_thresh: int, target_thresh: int,
                               utterances_thresh: int, speaker_thresh_indiv: int,
                               target_thresh_indiv: int, utterances_thresh_indiv: int,
                               utterance_thresh_func: Optional[Callable[[Tuple[Utterance, Utterance]], bool]]=None,
                               focus: str="speakers",
                               split_by_attribs: Optional[List[str]]=None,
                               speaker_attribs: Optional[Dict]=None,
                               target_attribs: Optional[Dict]=None) -> CoordinationScore:
        assert not isinstance(speakers, str)
        assert focus == "speakers" or focus == "targets"

        if split_by_attribs is None: split_by_attribs = []
        if speaker_attribs is None: speaker_attribs = {}
        if target_attribs is None: target_attribs = {}

        tally = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        cond_tally = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        cond_total = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        n_utterances = defaultdict(lambda: defaultdict(int))
        targets = defaultdict(set)
        real_speakers = set()
        for utt2 in utterances:
            if corpus.has_utterance(utt2.reply_to):
                speaker = utt2.speaker
                utt1 = corpus.get_utterance(utt2.reply_to)
                target = utt1.speaker
                if speaker == target: continue
                speaker, target = Coordination._annot_speaker(speaker, utt2, split_by_attribs), \
                                  Coordination._annot_speaker(target, utt1, split_by_attribs)

                speaker_has_attribs = Coordination._utterance_has_attribs(utt2, speaker_attribs)
                target_has_attribs = Coordination._utterance_has_attribs(utt1, target_attribs)

                if not speaker_has_attribs or not target_has_attribs: continue

                real_speakers.add(speaker)

                if utterance_thresh_func is None or \
                        utterance_thresh_func(utt2, utt1):
                    if focus == "targets": speaker, target = target, speaker
                    targets[speaker].add(target)
                    n_utterances[speaker][target] += 1
                    for cat in utt1.meta["liwc-categories"].union(utt2.meta["liwc-categories"]):
                        if cat in utt2.meta["liwc-categories"]:
                            tally[speaker][cat][target] += 1
                        if cat in utt1.meta["liwc-categories"]:
                            cond_total[speaker][cat][target] += 1
                            if cat in utt2.meta["liwc-categories"]:
                                cond_tally[speaker][cat][target] += 1

        out = CoordinationScore()
        if focus == "targets":
            speaker_thresh, target_thresh = target_thresh, speaker_thresh
            speaker_thresh_indiv, target_thresh_indiv = target_thresh_indiv, speaker_thresh_indiv
            real_speakers = list(targets.keys())

        for speaker in real_speakers:
            if speaker[0] not in speakers and focus != "targets": continue
            coord_w = {}  # coordination score wrt a category
            for cat in CoordinationWordCategories:
                threshed_cond_total = 0
                threshed_cond_tally = 0
                threshed_tally = 0
                threshed_n_utterances = 0
                for target in targets[speaker]:
                    if tally[speaker][cat][target] >= speaker_thresh_indiv and \
                            cond_total[speaker][cat][target] >= target_thresh_indiv and \
                            n_utterances[speaker][target] >= utterances_thresh_indiv:
                        threshed_cond_total += cond_total[speaker][cat][target]
                        threshed_cond_tally += cond_tally[speaker][cat][target]
                        threshed_tally += tally[speaker][cat][target]
                        threshed_n_utterances += n_utterances[speaker][target]
                if threshed_cond_total >= max(target_thresh, 1) and \
                    threshed_tally >= speaker_thresh and \
                    threshed_n_utterances >= max(utterances_thresh, 1):
                    coord_w[cat] = threshed_cond_tally / threshed_cond_total - \
                            threshed_tally / threshed_n_utterances
            if len(coord_w) > 0:
                out[speaker if split_by_attribs else speaker[0]] = coord_w
        return out
