import json
from collections import defaultdict

counts = defaultdict(int)
threads = defaultdict(set)
total = 0
with open("reddit-convos.json", "r") as f:
    l = json.load(f)
    for p in l:
        counts[p["speaker-info"]["subreddit"]] += 1
        threads[p["speaker-info"]["subreddit"]].add(p["root"])
        total += 1

print("subreddit, num comments, num threads")
for k, v in counts.items():
    print(k, v, len(threads[k]))

print()
print(total, "comments")
print(sum([len(v) for v in threads.values()]), "threads")
