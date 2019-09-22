from convokit.transformer import Transformer
from convokit.model import Corpus
from convokit.user_convo_helpers.user_convo_attrs import UserConvoAttrs
from convokit.user_convo_helpers.user_convo_utils import get_user_convo_attribute_table

from collections import Counter 
from itertools import chain
import numpy as np


class UserConvoDiversity(Transformer):
    '''
    implements methodology for calculating linguistic diversity from http://www.cs.cornell.edu/~cristian/Finding_your_voice__linguistic_development.html . 

    Outputs the following (user, convo) attributes:
        * self_div (within-diversity in paper)
        * other_div (across-diversity in paper)
        * adj_other_div (relative diversity in paper)
    Note that np.nan is returned for (user, convo) pairs with not enough text.

    :param stage_size: number of conversations per lifestage
    :param max_exp: highest experience level (i.e., # convos taken) to compute diversity scores for.
    :param sample_size: number of words to sample per convo
    :param min_convo_len: minimum number of utterances a user contributes per convo for that (user, convo) to get scored
    :param n_iters: number of samples to take for perplexity scoring
    :param cohort_delta: timespan between when users start for them to be counted as part of the same cohort. defaults to 2 months
    :param verbosity: amount of output to print
    :param test: if True, only runs for 1000 (user,convo)s. 
    '''
    
    def __init__(self, stage_size=20, max_exp=120, sample_size=200, min_convo_len=1,
                n_iters=50, cohort_delta=60*60*24*30*2, verbosity=250, test=False):
        self.stage_size = stage_size
        self.max_exp = max_exp
        self.sample_size = sample_size
        self.min_convo_len = 1
        self.n_iters = 50
        self.cohort_delta = cohort_delta
        self.verbosity = verbosity
        self.test = test
        if test:
            self.n_iters = 5
        
        
    def _chain_tokens(self, series):
        return list(chain(*series))
    
    def _nan_mean(self, arr):
        arr = [x for x in arr if not np.isnan(x)]
        if len(arr) > 0:
            return np.mean(arr)
        else:
            return np.nan
        
    def _perplexity(self, test_text, train_text):
        N_train, N_test = len(train_text), len(test_text)
        if min(N_train, N_test) == 0: return np.nan
        train_counts = Counter(train_text)
        return sum(
                -np.log(train_counts.get(tok, 1)/N_train) for tok in test_text
            )/N_test

    def _compute_divergences(self, cmp_samples, self_samples, 
                            other_samples, div_fn=None):
        if div_fn is None:
            div_fn = self._perplexity
        self_divs = [] 
        other_divs = []
        adj_other_divs = []

        for cmp_sample, self_sample, other_sample in zip(cmp_samples, self_samples, other_samples):

            self_divs.append(div_fn(cmp_sample, self_sample))
            other_divs.append(div_fn(cmp_sample, other_sample))


            adj_other_divs.append(other_divs[-1] - self_divs[-1])
        return {'self_div': self._nan_mean(self_divs), 
                'other_div': self._nan_mean(other_divs), 
                'adj_other_div': self._nan_mean(adj_other_divs)}
    
    def transform(self, corpus: Corpus):
        convos_per_split = self.stage_size // 2
        ref_sample_size = self.sample_size * convos_per_split
        
        if self.verbosity is not None:
            print('preparing corpus')
        join_tokens = UserConvoAttrs(attr_name='tokens', 
                     agg_fn=lambda x: ' '.join(x))
        corpus = join_tokens.fit_transform(corpus)
        text_df = get_user_convo_attribute_table(corpus, ['tokens','n_utterances'],
                            min_n_convos=self.stage_size, max_convo_idx=self.max_exp)
        text_df['stage_idx'] = (text_df.convo_idx // self.stage_size).map(int)

        user_df = text_df.drop_duplicates('user')\
            .set_index('user')[['user_start_time', 'user_n_convos']]
        
        text_df['tokenized'] = text_df.tokens.apply(lambda x: x.lower().split())
        text_df['wordcount'] = text_df.tokenized.apply(lambda x: len(x))
        
        ref_groups = text_df[text_df.convo_idx % 2 == 0].groupby(['user','stage_idx'])
        ref_df = ref_groups.tokenized.agg(self._chain_tokens).to_frame()\
            .join(ref_groups.wordcount.agg(sum))\
            .reset_index().join(user_df, on='user')
        ref_df = ref_df[ref_df.wordcount >= ref_sample_size]
        
        cmp_df = text_df[(text_df.convo_idx % 2 == 1)
                        & (text_df.n_utterances >= self.min_convo_len)
                        & (text_df.wordcount >= self.sample_size)
                        & (text_df.user_n_convos >= self.max_exp)]
        
        if self.test: cmp_df = cmp_df.head(1000)
        if self.verbosity is not None:
            print('computing diversities')
        for idx, (user_convo_id, row) in enumerate(cmp_df.iterrows()):
            if (self.verbosity is not None) \
                    and (idx % self.verbosity == 0) and (idx > 0):
                print(idx, '/', len(cmp_df))
            
            cmp_samples = np.random.choice(row.tokenized, (self.n_iters, self.sample_size))
            self_cmp = ref_df[(ref_df.user == row.user) & (ref_df.stage_idx == row.stage_idx)]
            if len(self_cmp) > 0:
                self_samples = np.random.choice(self_cmp.tokenized.values[0], 
                                                (self.n_iters, ref_sample_size))
            else:
                self_samples = [[]] * self.n_iters            
            
            other_cmp = ref_df[(ref_df.user != row.user) & (ref_df.stage_idx == row.stage_idx)
                      & (ref_df.user_n_convos >= (row.stage_idx + 1) * self.stage_size)]
            if self.cohort_delta is not None:
                other_cmp = other_cmp[other_cmp.user_start_time.between(
                    row.user_start_time - self.cohort_delta, row.user_start_time + self.cohort_delta)]
            if len(other_cmp) > 0:
                other_samples = [np.random.choice(tokens, ref_sample_size) 
                                 for tokens in other_cmp.tokenized.sample(self.n_iters, replace=True)]
            else:
                other_samples = [[]] * self.n_iters
            
            for k, v in self._compute_divergences(cmp_samples, self_samples, other_samples).items():
                corpus.get_user(row.user).meta['conversations'][row.convo_id][k] = v
        return corpus