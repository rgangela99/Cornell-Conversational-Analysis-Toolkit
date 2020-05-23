# original author: Tiffany Wang (tw292@cornell.edu)

import json
import os

location = os.getcwd()
subs_dict = {}
for file in os.listdir(os.path.join(location, 'reddit-data')):
    print(file)
    if file.endswith(".jsonlist"):
        with open(os.path.join(location, 'reddit-data', file), encoding="latin") as f:
            subs_dict[file[:-9]] = []
            for line in f:
                try:
                    line_json = json.loads(line)
                except json.decoder.JSONDecodeError:
                    print("decode error")
                    continue
                # remove duplicates
                #if line_json not in subs_dict[file[:-9]]:
                subs_dict[file[:-9]].append(line_json) 
                #if len(subs_dict[file[:-9]]) == 100: break

#deleted_idx = 0
def proc_author(s, root_id):
#    global deleted_idx
    if s == "[deleted]":
        s = "[deleted-{}]".format(root_id)
#        deleted_idx += 1
    return s

def make_user_info(d):
    return {k.replace("_", "-"): v for k, v in d.items() if k not in
        ["id", "link_id"]}

def children(comment):
    l = [{k: v for k, v in d.items() if k != "children"}
        for d in comment["children"]]
    for child in comment["children"]:
        l += children(child)
    return l

def del_dups(convos):
    """Delete duplicates in created convokit object.
    Parameter: Dataset in Convokit format.
    Returns: Deduplicated list"""
    ids = set()
    unique = []
    for x in convos:
        if x['id'] not in ids:
            ids.add(x['id'])
            unique.append(x)
    return unique

MIN_THREAD_LENGTH = 10
N_THREADS_PER_SUBREDDIT = 500

##create new json list
print("creating dataset")
reddit_convos = []
for subreddit in list(subs_dict.keys()):
    print(subreddit)
    n_good_threads = 0
    for self_post in reversed(subs_dict[subreddit]):
        for thread_root in sorted(self_post["children"],
            key=lambda x: x["created_utc"], reverse=True):
            if n_good_threads >= N_THREADS_PER_SUBREDDIT: break
            cs = children(thread_root)
            if len(cs) < MIN_THREAD_LENGTH - 1: continue
            n_good_threads += 1
            for child in cs:
                #if "body" not in child: continue
                if "body" not in child: child["body"] = "[deleted]"
                child.pop("name")
                child.pop("subreddit_id")
                d = {}
                d["root"] = thread_root["id"]
                d["id"] = child["id"]
                d["reply-to"] = child.pop("parent_id")
                d["text"] = child.pop("body")
                d["timestamp"] = int(child.pop("created_utc"))
                user_deleted = child["author"] == "[deleted]"
                d["speaker"] = proc_author(child.pop("author"), d["root"])
                d["speaker-info"] = make_user_info(child)
                d["speaker-info"]["self-post-id"] = self_post["id"]
                d["speaker-info"]["post-deleted"] = d["text"] == "[deleted]"
                d["speaker-info"]["speaker-deleted"] = user_deleted
                reddit_convos.append(d)
            d = {}
            del thread_root["name"], thread_root["subreddit_id"]
            del thread_root["parent_id"]
            d["root"] = thread_root["id"]
            d["id"] = thread_root.pop("id")
            d["reply-to"] = None
            d["text"] = thread_root.pop("body")
            d["timestamp"] = int(thread_root.pop("created_utc"))
            user_deleted = thread_root["author"] == "[deleted]"
            d["speaker"] = proc_author(thread_root.pop("author"), d["root"])
            thread_root.pop("children")
            d["speaker-info"] = make_user_info(thread_root)
            d["speaker-info"]["self-post-id"] = self_post["id"]
            d["speaker-info"]["post-deleted"] = d["text"] == "[deleted]"
            d["speaker-info"]["speaker-deleted"] = user_deleted

            reddit_convos.append(d)
        #for child in children(comment): #do some formatting
        #    if "body" not in child: continue
        #    child.pop('name') 
        #    #child.pop('children')
        #    child.pop('subreddit')
        #    child.pop('subreddit_id')
        #    d = {}
        #    d['root'] = comment['id']
        #    d['id'] = child['id']
        #    d['reply-to'] = child.pop('parent_id')
        #    d['text'] = child.pop('body')
        #    d['timestamp'] = int(child.pop('created_utc'))
        #    d['speaker'] = child.pop('author')
        #    d['speaker-info'] = child
        #    reddit_convos.append(d) #append to convokit
        #addcomment = {}#dict(comment)
        #addcomment['root'] = comment['id'] #cleanup comment
        #addcomment['id'] = comment.pop('id')
        #addcomment['text'] = comment.pop('selftext')
        #addcomment['timestamp'] = int(comment.pop('created_utc'))
        #addcomment['speaker'] = comment.pop('author')
        #comment.pop('children')
        #comment.pop('created')
        #addcomment['speaker-info'] = comment
        #reddit_convos.append(addcomment) #add the original comment
    del subs_dict[subreddit]

reddit_convos = del_dups(reddit_convos)
reddit_convos = [d for d in reddit_convos if "text" in d]

with open('reddit-convos.json', 'w') as fp:
    json.dump(reddit_convos, fp, indent=2)
