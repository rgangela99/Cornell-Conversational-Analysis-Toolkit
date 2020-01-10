import convokit
import numpy as np
import matplotlib.pyplot as plt

print("Loading corpus")
corpus = convokit.Corpus(filename=convokit.download("reddit-corpus-small"))

print("Computing hypergraph features")
hc = convokit.HyperConvo(prefix_len=10, include_root=False)
hc.fit_transform(corpus)

threads_feats = dict()
convos = corpus.iter_conversations()

for convo in convos:
    threads_feats.update(convo.meta['hyperconvo'])

feat_names = list(sorted(threads_feats[list(threads_feats.keys())[0]].keys()))

print("Computing low-dimensional embeddings")
te = convokit.ThreadEmbedder(return_components=True)
te.fit_transform(corpus)
X_threads = corpus.meta["threadEmbedder"]["X"]
roots = corpus.meta["threadEmbedder"]["roots"]
components = corpus.meta["threadEmbedder"]["components"]

ce = convokit.CommunityEmbedder(community_key="subreddit")
ce.fit_transform(corpus)
X_communities = corpus.meta["communityEmbedder"]["pts"]
subreddits = corpus.meta["communityEmbedder"]["labels"]

print("TOP THREADS")
for d in range(7):
    print("dimension {}".format(d))
    print("- most-negative threads")
    ranked = list(sorted(zip(roots, X_threads), key=lambda x: x[1][d]))
    for label, x in ranked[:10]:
        print("\t{}  {:.4f}".format(label, x[d]))
    print("- most-positive threads")
    for label, x in reversed(ranked[-10:]):
        print("\t{}  {:.4f}".format(label, x[d]))
    print()
    print()

print("TOP SUBREDDITS")
for d in range(7):
    print("dimension {}".format(d))
    print("- most-negative subreddits")
    ranked = list(sorted(zip(subreddits, X_communities), key=lambda x: x[1][d]))
    for label, x in ranked[:10]:
        print("\t{}  {:.4f}".format(label, x[d]))
    print("- most-positive subreddits")
    for label, x in reversed(ranked[-10:]):
        print("\t{}  {:.4f}".format(label, x[d]))
    print()
    print()

print()
print("TOP FEATURES")
for d in range(7):
    print("dimension {}".format(d))
    print("- most negative features")
    ranked = list(sorted(zip(feat_names, np.transpose(components)),
        key=lambda x: x[1][d]))
    for label, x in ranked[:10]:
        print("\t{}  {:.4f}".format(label, x[d]))
    print("- most positive features")
    for label, x in reversed(ranked[-10:]):
        print("\t{}  {:.4f}".format(label, x[d]))
    print()
    print()
