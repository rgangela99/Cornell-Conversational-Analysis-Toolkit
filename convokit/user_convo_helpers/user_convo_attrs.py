from convokit.transformer import Transformer
from convokit.model import Corpus

class UserConvoAttrs(Transformer):

    '''
        Transformer that aggregates statistics per (user, convo). e.g., average wordcount of all utterances that user contributed per convo. Assumes that `corpus.organize_user_convo_history` has already been called.

        :param attr_name: name of attribute to aggregate over. note that this attribute must already exist as an annotation to utterances in the corpus.
        :param output_field: name of the aggregated attribute to output. defaults to `attr_name`.
        :param agg_fn: function to aggregate utterance-level attribute with. defaults to returning a list.
        :param recompute: if `False`, will not recompute the aggregate if `output_field` already exists for a user convo entry.
    '''
    
    def __init__(self, attr_name, output_field=None, agg_fn=None, recompute=False):
        self.attr_name = attr_name
        if output_field is None:
            self.output_field = attr_name
        else:
            self.output_field = output_field
        if agg_fn is None:
            self.agg_fn = list
        else:
            self.agg_fn = agg_fn
        self.recompute = recompute
    
    def transform(self, corpus: Corpus):
        '''
            creates and populates user, convo aggregates.

            :param corpus: the Corpus to transform.
            :type corpus: Corpus  
        '''

        for user in corpus.iter_users():
            if 'conversations' not in user.meta: continue

            for convo_id, convo in user.meta['conversations'].items():
                if self.recompute or (corpus.get_user_convo_info(user.name, convo_id, self.output_field) is None):
                    utterance_attrs = [corpus.get_utterance(utt_id).meta[self.attr_name] 
                                       for utt_id in convo['utterance_ids']]
                    corpus.set_user_convo_info(user.name, convo_id, self.output_field, self.agg_fn(utterance_attrs))
        return corpus