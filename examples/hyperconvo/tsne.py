import convokit
import numpy as np
import matplotlib.pyplot as plt

print("Loading corpus")
corpus = convokit.Corpus(filename=convokit.download("reddit-corpus-small"))

print("Computing hypergraph features")
hc = convokit.HyperConvo()
hc.fit_transform(corpus)

print("Computing low-dimensional embeddings")
te = convokit.ThreadEmbedder(n_components=7)
te.fit_transform(corpus)

ce = convokit.CommunityEmbedder(community_key="subreddit", method="tsne")
ce.fit_transform(corpus)

pts = corpus.meta["communityEmbedder"]["pts"]
labels = corpus.meta["communityEmbedder"]["labels"]

xs, ys = zip(*pts)
plt.scatter(xs, ys)
for i, txt in enumerate(labels):
    plt.annotate(txt, (xs[i], ys[i]))
plt.savefig("tsne")
plt.show()
