import pandas as pd 
import numpy as np

def get_user_convo_attribute_table(corpus, attrs, min_n_convos=0, max_convo_idx=None):
    '''
        returns a table where each row lists a (user, convo) level aggregate for each attribute in attrs. for convenience in later analyses, also returns the total # of convos and the timestamp of the first convo for each user.

        :param corpus: corpus to extract user, convo info from
        :param attrs: list of (user, convo) attribute names
        :param min_n_convos: minimum number of convos a user participates in for their statistics to be returned
        :param max_convo_idx: max index of convo to return per user
    '''
    attr_entries = []
    for user in corpus.iter_users():
        if 'conversations' not in user.meta: continue
        if user.meta['n_convos'] < min_n_convos: continue
        
        for convo_id, convo_dict in user.meta['conversations'].items():
            if (max_convo_idx is not None) and (convo_dict['idx'] >= max_convo_idx): continue
            entry = {'key': '%s__%s' % (user.name, convo_id),
                    'user': user.name, 'convo_id': convo_id, 'convo_idx': convo_dict['idx']}
            for attr in attrs:
                entry[attr] = convo_dict.get(attr, None)
            entry['user_n_convos'] = user.meta['n_convos']
            entry['user_start_time'] = user.meta['start_time']
            attr_entries.append(entry)
    return pd.DataFrame(attr_entries).set_index('key')

def get_lifestage_attributes(attr_df, attr, lifestage_size, min_n_convos, agg_fn=np.mean):
    '''
        returns a table aggregating attribute per user lifestage.

        :param attr_df: table containing (user, convo)-level attribute
        :param attr: attribute to aggregate
        :param lifestage_size: # of convos per lifestage
        :param min_n_convos: minimum number of convos a user participates in for their statistics to be returned
        :param agg_fn: function to aggregate attribute with, defaults to mean.
    '''
    subset = attr_df[(attr_df.user_n_convos >= min_n_convos)
                    & (attr_df.convo_idx < min_n_convos)]
    aggregates = subset.groupby(['user', subset.convo_idx // lifestage_size])[attr].agg(agg_fn)
    aggregates = aggregates.reset_index().pivot(index='user', columns='convo_idx', values=attr)
    return aggregates