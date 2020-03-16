graphics_dir = "graphics"

def get_graphic_from_feature(feat):
    stat = feat.split("[")[0]
    midthread = "mid-thread" in feat
    base_feat = feat.split("[")[1].split("]")[0].replace("mid-thread ", "").replace(" over mid-thread", "")

    img_id = str(abs(hash(base_feat)))[:5]

# {x: str(abs(hash(x)))[:5] for x in set([b.replace(" ", "_").replace("->", "") for b in base if 'mid-thread' not in b])}

# {'outdegree_over_Cc_responses': '35387',
#  'external_reciprocity_motif': '55320',
#  'outgoing_triads': '58453',
#  'indegree_over_cc_responses': '33180',
#  'outdegree_over_cc_responses': '60556',
#  'incoming_triads': '46587',
#  'reciprocity_motif': '15176',
#  'indegree_over_Cc_responses': '27366',
#  'indegree_over_CC_responses': '61676',
#  'dyadic_interaction_motif': '49039',
#  'outdegree_over_CC_responses': '16882'}

