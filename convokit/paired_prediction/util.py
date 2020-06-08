from random import choice, shuffle
from pandas import DataFrame
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from convokit.classifier.util import extract_feats_from_obj


def generate_bow_paired_X_y(pair_orientation_feat_name, pair_id_to_objs, vector_name):
    """
    Generate the X, y matrix for paired prediction
    :param pair_id_to_objs: dictionary indexed by the paired feature instance value, with the value
    being a tuple (pos_obj, neg_obj)
    :return: X, y matrix representing the predictive features and labels respectively
    """
    pos_obj_dict = dict()
    neg_obj_dict = dict()
    for pair_id, (pos_obj, neg_obj) in pair_id_to_objs.items():
        pos_obj_dict[pair_id] = pos_obj.meta[vector_name]
        neg_obj_dict[pair_id] = neg_obj.meta[vector_name]

    X, y = [], []
    pair_ids = list(pair_id_to_objs)
    shuffle(pair_ids)
    for pair_id in pair_ids:
        pos_feats = pos_obj_dict[pair_id]
        neg_feats = neg_obj_dict[pair_id]
        orientation = pair_id_to_objs[pair_id][0].meta[pair_orientation_feat_name]

        assert orientation in ["pos", "neg"]
        if orientation == "pos":
            y.append(1)
            diff = pos_feats - neg_feats
        else:
            y.append(0)
            diff = neg_feats - pos_feats

        X.append(diff)

    if issparse(X[0]): # for csr_matrix
        X = vstack(X)
    else: # for non-compressed numpy arrays
        X = np.vstack(X)

    return X, np.array(y)


def generate_paired_X_y(pred_feats, pair_orientation_feat_name, pair_id_to_objs):
    """
    Generate the X, y matrix for paired prediction
    :param pair_id_to_objs: dictionary indexed by the paired feature instance value, with the value
    being a tuple (pos_obj, neg_obj)
    :return: X, y matrix representing the predictive features and labels respectively
    """
    pos_obj_dict = dict()
    neg_obj_dict = dict()
    for pair_id, (pos_obj, neg_obj) in pair_id_to_objs.items():
        pos_obj_dict[pair_id] = extract_feats_from_obj(pos_obj, pred_feats)
        neg_obj_dict[pair_id] = extract_feats_from_obj(neg_obj, pred_feats)
    pos_obj_df = DataFrame.from_dict(pos_obj_dict, orient='index')
    neg_obj_df = DataFrame.from_dict(neg_obj_dict, orient='index')

    X, y = [], []
    pair_ids = list(pair_id_to_objs)
    shuffle(pair_ids)
    for pair_id in pair_ids:
        pos_feats = np.array(pos_obj_df.loc[pair_id]).astype('float64')
        neg_feats = np.array(neg_obj_df.loc[pair_id]).astype('float64')
        orientation = pair_id_to_objs[pair_id][0].meta[pair_orientation_feat_name]

        assert orientation in ["pos", "neg"]

        if orientation == "pos":
            y.append(1)
            diff = pos_feats - neg_feats
        else:
            y.append(0)
            diff = neg_feats - pos_feats

        X.append(diff)

    return csr_matrix(np.array(X)), np.array(y)

def generate_pair_id_to_objs(corpus, obj_type, selector, pair_orientation_feat_name, label_feat_name, pair_id_feat_name):
    pair_id_to_obj = {'pos': dict(), 'neg': dict()}
    for obj in corpus.iter_objs(obj_type, selector):
        if obj.meta[pair_orientation_feat_name] is None: continue
        pair_id_to_obj[obj.meta[label_feat_name]][obj.meta[pair_id_feat_name]] = obj

    pair_ids = set(pair_id_to_obj['pos'].keys()).intersection(set(pair_id_to_obj['neg'].keys()))

    # print(set(pair_id_to_obj['pos'].keys()))
    print("Found {} valid pairs.".format(len(pair_ids)))
    pair_id_to_objs = dict()
    for pair_id in pair_ids:
        pair_id_to_objs[pair_id] = (pair_id_to_obj['pos'][pair_id], pair_id_to_obj['neg'][pair_id])
    return pair_id_to_objs