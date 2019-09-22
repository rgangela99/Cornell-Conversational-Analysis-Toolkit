from convokit.transformer import Transformer
from convokit.model import Corpus

class UserConvoAttrs(Transformer):

    '''
        Transformer that aggregates statistics per (user, convo). e.g., average wordcount of all utterances that user contributed per convo.

        :param attr_name: name of attribute to aggregate over. note that this attribute must already exist as an annotation to utterances in the corpus.
        :param agg_fn: function to aggregate utterance-level attribute with. defaults to returning a list.
    '''
    
    def __init__(self, attr_name, agg_fn=None):
        self.attr_name = attr_name
        if agg_fn is None:
            self.agg_fn = list
        else:
            self.agg_fn = agg_fn
    
    def transform(self, corpus: Corpus):
        for user in corpus.iter_users():
            if 'conversations' not in user.meta: continue

            for convo_id, convo in user.meta['conversations'].items():
                utterance_attrs = [corpus.get_utterance(utt_id).meta[self.attr_name] 
                                   for utt_id in convo['utterance_ids']]
                user.meta['conversations'][convo_id][self.attr_name] = self.agg_fn(utterance_attrs)
        return corpus