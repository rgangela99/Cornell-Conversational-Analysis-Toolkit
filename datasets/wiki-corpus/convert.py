# convert into toolkit json format
import json
import time
import datetime

MaxUtterances = -1

KeyId = "id"
KeyUser = "speaker"
KeyConvoRoot = "root"
KeyReplyTo = "reply-to"
KeyTimestamp = "timestamp"
KeyText = "text"
KeyUserInfo = "meta"

genders = {}
edit_counts = {}
with open("wikipedia.talkpages.userinfo.txt", "r") as f:
    for line in f:
        name, edit_count, gender, uid = line.strip().split(" +++$+++ ")
        genders[name] = gender
        edit_counts[name] = edit_count

with open("wikipedia.talkpages.admins.txt", "r", encoding="utf-8") as f:
    admins = {}
    for line in f:
        toks = line.strip().split(" ")
        name = " ".join(toks[:-1])
        date = toks[-1]
        if date == "NA": date = "1975-01-01"
        admins[name] = time.mktime(
            datetime.datetime.strptime(date.strip(), "%Y-%m-%d").timetuple())

users_meta = {}
uniq_admins = set()
usernames = set()
usernames_cased = set()
unrecoverable_replytos = 0
with open("wikipedia.talkpages.conversations.dat", "r", encoding="utf-8") as f:
    utterances = []
    count = 0
    unk_count = 0
    for line in f:
        if count % 1000 == 0: print(count)
        line = line[:-1]
        if line:
            fields = line.split(" +++$+++ ")
            if len(fields) == 9:
                user = fields[1].strip()
                if user == "":
                    user = "{unknown-" + str(unk_count) + "}"
                    unk_count += 1
                timestamp = fields[6]
                is_admin = False
                if user in admins and float(timestamp) > admins[user]:
                    is_admin = True
                    #speaker += "{admin}"
                    uniq_admins.add(user)

                is_admin_glob = is_admin
                if user in users_meta and users_meta[user]["is-admin"]:
                    is_admin_glob = True
                    
                users_meta[user] = {
                    "is-admin": is_admin_glob,
                    "gender": gender[user] if user in gender else "unknown",
                    "edit-count": edit_counts[user] if user in \
                            edit_counts else "unknown"
                }

                d = {
                    KeyId: fields[0],
                    KeyUser: user,
                    KeyConvoRoot: fields[3],
                    KeyTimestamp: timestamp,
                    KeyText: fields[7],
                    KeyUserInfo: {
                        "is-admin": is_admin
                    }
                }
                fields[4] = fields[4].strip()
                if fields[4] != "initial_post" and fields[4] != "-1":
                    d[KeyReplyTo] = fields[4]
                if fields[4] == "-1":
                    unrecoverable_replytos += 1
                usernames.add(fields[1].strip())
                usernames_cased.add(fields[1].strip())
                utterances.append(d)
                count += 1
                #if MaxUtterances > 0 and count > MaxUtterances:
                #    break

#udict = {u["id"]: u for u in utterances}
#for i, u in enumerate(utterances):
#    if KeyReplyTo in u:
#        target = udict[u[KeyReplyTo]][KeyUser]
#        #u[KeyConvoId] = u[KeyUser] + "->" + (
#        #    "{admin}" if target.endswith("{admin}") else "{nonadmin}")  # target groups
#        #u[KeyConvoId] = target  # target groups -- experimental
#        u[KeyConvoId] = u[KeyUser]  # speaker groups
#        utterances[i] = u
#    else:
#        del utterances[i][KeyConvoId]

ips = set()
for user in usernames:
    if user and user[0] in "0123456789":
        ips.add(user)
print(len(ips), len(usernames))

if MaxUtterances > 0:
    #import random
    #random.shuffle(utterances)
    utterances = utterances[-MaxUtterances:]
json.dump(utterances, open("utterances.json", "w"), indent=2,
          sort_keys=True)

with open("speakers.json", "w") as f:
    json.dump(users_meta, f, indent=2)

print(len(uniq_admins), "admins")
print(unrecoverable_replytos, "unrecovered reply-tos")
#print(len(usernames), len(usernames_cased))
print("Done")

