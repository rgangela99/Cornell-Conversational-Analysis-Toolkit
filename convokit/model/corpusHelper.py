import os
import json
from collections import defaultdict
import pickle
from .user import User
from .utterance import Utterance
from .conversation import Conversation
from typing import Dict

BIN_DELIM_L, BIN_DELIM_R = "<##bin{", "}&&@**>"
KeyId = "id"
KeyUser = "user"
KeyConvoRoot = "root"
KeyReplyTo = "reply-to"
KeyTimestamp = "timestamp"
KeyText = "text"
DefinedKeys = {KeyId, KeyUser, KeyConvoRoot, KeyReplyTo, KeyTimestamp, KeyText}
KeyMeta = "meta"

def load_uttinfo_from_dir(dirname, utterance_start_index, utterance_end_index, exclude_utterance_meta):
    assert dirname is not None
    assert os.path.isdir(dirname)

    if utterance_start_index is None: utterance_start_index = 0
    if utterance_end_index is None: utterance_end_index = float('inf')

    if os.path.exists(os.path.join(dirname, 'utterances.jsonl')):
        with open(os.path.join(dirname, 'utterances.jsonl'), 'r') as f:
            utterances = []
            idx = 0
            for line in f:
                if utterance_start_index <= idx <= utterance_end_index:
                    utterances.append(json.loads(line))
                idx += 1

    elif os.path.exists(os.path.join(dirname, 'utterances.json')):
        with open(os.path.join(dirname, "utterances.json"), "r") as f:
            utterances = json.load(f)
            utterances = utterances[utterance_start_index:min(len(utterances), utterance_end_index+1)]

    if exclude_utterance_meta:
        for utt in utterances:
            for field in exclude_utterance_meta:
                del utt["meta"][field]

    return utterances


def load_users_meta_from_dir(filename, exclude_user_meta):
    with open(os.path.join(filename, "users.json"), "r") as f:
        users_meta = defaultdict(dict)
        for k, v in json.load(f).items():
            if k in exclude_user_meta: continue
            users_meta[k] = v
    return users_meta


def load_convos_meta_from_dir(filename, exclude_conversation_meta):
    with open(os.path.join(filename, "conversations.json"), "r") as f:
        convos_meta = defaultdict(dict)
        for k, v in json.load(f).items():
            if k in exclude_conversation_meta: continue
            convos_meta[k] = v
    return convos_meta


def load_corpus_meta_from_dir(filename, corpus_meta, exclude_overall_meta):
    """
    Updates corpus meta object with fields from corpus.json
    """
    with open(os.path.join(filename, "corpus.json"), "r") as f:
        for k, v in json.load(f).items():
            if k in exclude_overall_meta: continue
            corpus_meta[k] = v


def unpack_binary_data_for_utts(utterances, filename, utterance_index, exclude_meta, KeyMeta):
    for field, field_type in utterance_index.items():
        if field_type == "bin" and field not in exclude_meta:
            with open(os.path.join(filename, field + "-bin.p"), "rb") as f:
                l_bin = pickle.load(f)
            for i, ut in enumerate(utterances):
                for k, v in ut[KeyMeta].items():
                    if k == field and type(v) == str and v.startswith(BIN_DELIM_L) and \
                            v.endswith(BIN_DELIM_R):
                        idx = int(v[len(BIN_DELIM_L):-len(BIN_DELIM_R)])
                        utterances[i][KeyMeta][k] = l_bin[idx]
    for field in exclude_meta:
        del utterance_index[field]


def unpack_binary_data(filename, obj_meta, object_index, obj_type, exclude_meta):
    """
    Unpack binary data for Users or Conversations
    """
    # unpack user meta
    for field, field_type in object_index.items():
        if field_type == "bin" and field not in exclude_meta:
            with open(os.path.join(filename, field + "-{}-bin.p".format(obj_type)), "rb") as f:
                l_bin = pickle.load(f)
            for user, metadata in obj_meta.items():
                for k, v in metadata.items():
                    if k == field and type(v) == str and str(v).startswith(BIN_DELIM_L) and \
                            str(v).endswith(BIN_DELIM_R):
                        idx = int(v[len(BIN_DELIM_L):-len(BIN_DELIM_R)])
                        metadata[k] = l_bin[idx]
    for field in exclude_meta:
        del object_index[field]


def load_from_utterance_file(filename, utterance_start_index, utterance_end_index):
    """
    where filename is "utterances.json" or "utterances.jsonl" for example
    """
    with open(filename, "r") as f:
        try:
            ext = filename.split(".")[-1]
            if ext == "json":
                utterances = json.load(f)
            elif ext == "jsonl":
                utterances = []
                if utterance_start_index is None: utterance_start_index = 0
                if utterance_end_index is None: utterance_end_index = float('inf')
                idx = 0
                for line in f:
                    if utterance_start_index <= idx <= utterance_end_index:
                        utterances.append(json.loads(line))
                    idx += 1
        except Exception as e:
            raise Exception("Could not load corpus. Expected json file, encountered error: \n" + str(e))
    return utterances

def initialize_users_and_utterances_objects(corpus, utt_dict, utterances, users_dict, users_meta):
    """
    Initialize User and Utterance objects
    """
    for i, u in enumerate(utterances):
        u = defaultdict(lambda: None, u)
        user_key = u[KeyUser]
        if user_key not in users_dict:
            users_dict[user_key] = User(owner=corpus, id=u[KeyUser], meta=users_meta[u[KeyUser]])

        user = users_dict[user_key]

        # temp fix for reddit reply_to
        if "reply_to" in u:
            reply_to_data = u["reply_to"]
        else:
            reply_to_data = u[KeyReplyTo]
        utt = Utterance(owner=corpus, id=u[KeyId], user=user,
                       root=u[KeyConvoRoot],
                       reply_to=reply_to_data, timestamp=u[KeyTimestamp],
                       text=u[KeyText], meta=u[KeyMeta])

        utt_dict[utt.id] = utt

def merge_utterance_lines(utt_dict):
    """
    For merging adjacent utterances by the same user
    """
    new_utterances = {}
    for uid, utt in utt_dict.items():
        merged = False
        if utt.reply_to is not None and utt.user is not None:
            u0 = utt_dict[utt.reply_to]
            if u0.root == utt.root and u0.user == utt.user:
                new_utterances[u0.id].text += " " + utt.text
                merged = True
        if not merged:
            new_utterances[utt.id] = utt
    return new_utterances

def initialize_conversations(corpus, utt_dict, convos_meta):
    """
    Initialize Conversation objects from utterances and conversations metadata
    """
    # organize utterances by conversation
    convo_to_utts = defaultdict(list) # temp container identifying utterances by conversation
    for u in utt_dict.values():
        convo_key = u.root # each root is considered a separate conversation
        convo_to_utts[convo_key].append(u.id)
    conversations = {}
    for convo_id in convo_to_utts:
        # look up the metadata associated with this conversation, if any
        convo_meta = convos_meta.get(convo_id, None)
        convo = Conversation(owner=corpus, id=convo_id,
                             utterances=convo_to_utts[convo_id],
                             meta=convo_meta)
        conversations[convo_id] = convo
    return conversations

def dump_helper_bin(d: Dict, d_bin: Dict, fields_to_skip=None) -> Dict: # object_idx
    """

    :param d: The dict to encode
    :param d_bin: The dict of accumulated lists of binary attribs
    :param object_idx:
    :return:
    """
    if fields_to_skip is None:
        fields_to_skip = []

    d_out = {}
    for k, v in d.items():
        if k in fields_to_skip: continue
        try:   # try saving the field
            json.dumps(v)
            d_out[k] = v
            # if k not in object_idx:
            #     object_idx[k] = str(type(v))
        except (TypeError, OverflowError):   # unserializable
            d_out[k] = "{}{}{}".format(BIN_DELIM_L, len(d_bin[k]), BIN_DELIM_R)
            d_bin[k].append(v)
            # object_idx[k] = "bin"   # overwrite non-bin type annotation if necessary
    return d_out


def dump_corpus_object(corpus, dir_name, filename, obj_type, bin_name, fields_to_skip):
    with open(os.path.join(dir_name, filename), "w") as f:
        d_bin = defaultdict(list)
        objs = {u: dump_helper_bin(corpus.get_object(obj_type, u).meta, d_bin,
                                    fields_to_skip.get(obj_type, [])) for u in corpus.get_object_ids(obj_type)}
        json.dump(objs, f)

        for name, l_bin in d_bin.items():
            with open(os.path.join(dir_name, name + "-{}-bin.p".format(bin_name)), "wb") as f_pk:
                pickle.dump(l_bin, f_pk)

def dump_utterances(corpus, dir_name, fields_to_skip):
    with open(os.path.join(dir_name, "utterances.jsonl"), "w") as f:
        d_bin = defaultdict(list)

        for ut in corpus.iter_utterances():
            ut_obj = {
                KeyId: ut.id,
                KeyConvoRoot: ut.root,
                KeyText: ut.text,
                KeyUser: ut.user.id,
                KeyMeta: dump_helper_bin(ut.meta, d_bin, fields_to_skip.get('utterance', [])),
                KeyReplyTo: ut.reply_to,
                KeyTimestamp: ut.timestamp
            }
            json.dump(ut_obj, f)
            f.write("\n")

        for name, l_bin in d_bin.items():
            with open(os.path.join(dir_name, name + "-bin.p"), "wb") as f_pk:
                pickle.dump(l_bin, f_pk)