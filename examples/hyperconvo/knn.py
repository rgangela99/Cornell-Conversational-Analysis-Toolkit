import convokit
from sklearn.neighbors import NearestNeighbors

print("Loading corpus")
corpus = convokit.Corpus(filename=convokit.download("reddit-corpus-small"))

print("Computing hypergraph features")
hc = convokit.HyperConvo(prefix_len=10, include_root=False)
hc.fit_transform(corpus)

print("Computing low-dimensional embeddings")
te = convokit.ThreadEmbedder(n_components=7)
te.fit_transform(corpus)

ce = convokit.CommunityEmbedder(community_key="subreddit")
ce.fit_transform(corpus)

X_communities = corpus.meta["communityEmbedder"]["pts"]
subreddits = corpus.meta["communityEmbedder"]["labels"]

knn = NearestNeighbors(n_neighbors=10)
knn.fit(X_communities)

print("Nearest neighbors for each subreddit:")
for x, subreddit in zip(X_communities, subreddits):
    print(subreddit, "->", end=" ")
    for idx in knn.kneighbors([x], return_distance=False)[0][1:]:
        print(subreddits[idx], end=" ")
    print()
