from collections import defaultdict

graphics_dir = "graphics"

img_id_dict =  {'outdegree_over_Cc_responses': 'hTon_outdegree.png',
                 'external_reciprocity_motif': 'external_reciprocity.png',
                 'outgoing_triads': 'outgoing.png',
                 'indegree_over_cc_responses': 'nTon_indegree.png',
                 'outdegree_over_cc_responses': 'nTon_outdegree.png',
                 'incoming_triads': 'incoming.png',
                 'reciprocity_motif': 'reciprocity.png',
                 'indegree_over_Cc_responses': 'hTon_indegree.png',
                 'indegree_over_CC_responses': 'hToh_indegree.png',
                 'dyadic_interaction_motif': 'dyadic.png',
                 'outdegree_over_CC_responses': 'hToh_outdegree.png'}

def get_graphic_from_feature(feat):
    stat = feat.split("[")[0]
    midthread = "mid-thread" in feat
    base_feat = feat.split("[")[1].split("]")[0].replace("mid-thread ", "").replace(" over mid-thread", "")
    base_feat = base_feat.replace('->', "").replace(" ", '_')
    img_id = img_id_dict[base_feat]
    return img_id, midthread, stat

def get_graphic_dict(feats):
    graphic_triples = [get_graphic_from_feature(feat) for feat in feats]
    retval = defaultdict(list)
    for (img_id, midthread, stat) in graphic_triples:
        retval[(img_id, midthread)].append(stat)
    return retval

# {x: str(abs(hash(x)))[:5] for x in set([b.replace(" ", "_").replace("->", "") for b in base if 'mid-thread' not in b])}




