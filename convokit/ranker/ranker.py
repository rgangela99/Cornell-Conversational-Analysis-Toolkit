from convokit.model import Corpus, Conversation, User, Utterance
from typing import List, Callable, Union
from convokit import Transformer, CorpusObject
import pandas as pd

class Ranker(Transformer):
    def __init__(self, obj_type: str,
                 score_func: Callable[[CorpusObject], Union[int, float]],
                 selector: Callable[[CorpusObject], bool] = lambda obj: True,
                 score_feat_name: str = "ranking_score", rank_feat_name: str = "rank"):
        """

        :param obj_type: type of Corpus object to rank: 'conversation', 'user', or 'utterance'
        :param score_func: function for computing the score of a given object
        :param selector: function to select for Corpus objects to transform
        :param score_feat_name: metadata feature name to use in annotation for score value, default: "ranking_score"
        :param rank_feat_name: metadata feature name to use in annotation for the rank value, default: "rank"
        """
        self.obj_type = obj_type
        self.score_func = score_func
        self.score_feat_name = score_feat_name
        self.rank_feat_name = rank_feat_name
        self.selector = selector

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Annotate corpus objects with scores and rankings
        :param corpus: target corpus
        :return: annotated corpus
        """
        obj_iters = {"conversation": corpus.iter_conversations,
                     "user": corpus.iter_users,
                     "utterance": corpus.iter_utterances}
        obj_scores = [(obj.id, self.score_func(obj)) for obj in obj_iters[self.obj_type](self.selector)]
        df = pd.DataFrame(obj_scores, columns=["id", self.score_feat_name]) \
            .set_index('id').sort_values(self.score_feat_name, ascending=False)
        df[self.rank_feat_name] = [idx+1 for idx, _ in enumerate(df.index)]

        for obj in corpus.iter_objs(obj_type=self.obj_type):
            if obj.id in df.index:
                obj.add_meta(self.score_feat_name, df.loc[obj.id][self.score_feat_name])
                obj.add_meta(self.rank_feat_name, df.loc[obj.id][self.rank_feat_name])
            else:
                obj.add_meta(self.score_feat_name, None)
                obj.add_meta(self.rank_feat_name, None)
        return corpus

    def transform_objs(self, objs: List[CorpusObject]):
        """
        Annotate list of Corpus objects with scores and rankings
        :param objs: target list of Corpus objects
        :return: list of annotated COrpus objects
        """
        obj_scores = [(obj.id, self.score_func(obj)) for obj in objs]
        df = pd.DataFrame(obj_scores, columns=["id", self.score_feat_name]) \
            .set_index('id').sort_values(self.score_feat_name, ascending=False)
        df[self.rank_feat_name] = [idx+1 for idx, _ in enumerate(df.index)]
        for obj in objs:
            obj.add_meta(self.score_feat_name, df.loc[obj.id][self.score_feat_name])
            obj.add_meta(self.rank_feat_name, df.loc[obj.id][self.rank_feat_name])
        return objs

    def summarize(self, corpus: Corpus = None, objs: List[CorpusObject] = None):
        """
        Generate a dataframe indexed by object id, containing score + rank, and sorted by rank (in ascending order)
        of the objects in an annotated corpus, or a list of corpus objects
        :param corpus: annotated target corpus
        :param objs: list of annotated corpus objects
        :return: a pandas DataFrame
        """
        if ((corpus is None) and (objs is None)) or ((corpus is not None) and (objs is not None)):
            raise ValueError("summarize() takes in either a Corpus or a list of users / utterances / conversations")

        if objs is None:
            obj_iters = {"conversation": corpus.iter_conversations,
                         "user": corpus.iter_users,
                         "utterance": corpus.iter_utterances}
            obj_scores_ranks = [(obj.id, obj.meta[self.score_feat_name], obj.meta[self.rank_feat_name])
                          for obj in obj_iters[self.obj_type](self.selector)]
        else:
            obj_scores_ranks = [(obj.id, obj.meta[self.score_feat_name], obj.meta[self.rank_feat_name]) for obj in objs]

        df = pd.DataFrame(obj_scores_ranks, columns=["id", self.score_feat_name, self.rank_feat_name])\
                        .set_index('id').sort_values(self.rank_feat_name, ascending=True)

        return df
