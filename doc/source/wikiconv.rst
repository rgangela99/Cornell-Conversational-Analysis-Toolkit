Cornell Wikiconv Dataset
========================
WikiConv is a multilingual corpus encompassing the history of conversations on Wikipedia Talk Pages—including the deletion, modification and restoration of comments. More information is available at: https://github.com/conversationai/wikidetox/tree/master/wikiconv. 
Distributed together with `WikiConv A Corpus of the Complete Conversational History of a Large Online Collaborative Community <Http://www.cs.cornell.edu/~cristian/index_files/wikiconv-conversation-corpus.pdf>`_. 
Yiqing Hua, Cristian Danescu-Niculescu-Mizil, Dario Taraborelli, Nithum Thain, Jeffery Sorensen, Lucas Dixon. EMNLP 2018. 

Summary
-------
Organize the conversations of the Wikiconv Dataset to mirror the most up-to-date version of the conversation at the time of data collection. This most up-to-date version of the conversation reflects the "final" state of each utterance (the utterance after all edits at time of data collection have been performed on the utterance) in the conversation. 


Dataset Details
---------------

Types
^^^^^
Due to Wikipedia's format and editing style, conversations are just snapshots of a history of edits (or revisions) that take place on a given Talk Page. We parse these revisions, compute diffs, and categorize edits into 5 kinds of 'actions' and each utterance is made up of a collection of the following actions:

- CREATION: An edit that creates a new section in wiki markup.
- ADDITION: An edit that adds a new comment to the thread of a conversation.
- MODIFICATION: An edit that modifies an existing comment on a Talk Page.
- DELETION: An edit that removes a comment from a Talk Page.
- RESTORATION: An edit (for example, a revert) that restores a previously removed comment.

Utterance-level Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each conversational turn on the talk page is viewed as an utterance. For each utterance, we provide:

- id: A unique identifier assigned by the original WikiConv reconstruction pipeline
- user: Wikipedia user name of the editor
- root: ID of the conversation root of the utterance
- reply_to: ID of the utterance to which this utterance replies to (None if the utterance is not a reply)
- timestamp: Time of the utterance
- text: Textual content of the utterance
- type: Type of the action that this comment represents (from one of the five previously defined types)
- meta: Provided below

Metadata for each utterance includes:

- is_section_header: Whether the utterance is a conversation "title" or "subject" as seen on the original talk page (if true, this utterance should be ignored when doing any NLP tasks)
- indentation: Level of indentation of the comment when displayed on the page
- ancestor_id: For modification, removal, and restoration actions, this provides the id of first creation of the action that was modified, removed, or restored respectively.
- rev_id: The Wikipedia revision id of the edit from which the action was extracted.
- parent_id: For modification, removal, and restoration actions, this provides the id of the action that was modified, removed, or restored respectively.
- toxicity: Score assigned by Perspective API given the content using TOXICITY attribute (only available for English corpus).
- sever_toxicity: Score given by Perspective API given the content using SEVERE_TOXICITY attribute (only available for English corpus).
- original: Original Utterance if the original utterance was modified or deleted in any way 
- modification: A list of utterances indicating modification edits on this utterance, ordered by timestamp (earliest first)
- deletion: A list of utterances (of size 1 unless the utterance was restored and then deleted again) indicating deletion edits on this utterance, ordered by timestamp (earliest first)
- restoration: A list of utterances (of size 1 unless the utterance was deleted, restored, deleted and restored again) indicating restoration edits on this utterance, ordered by timestamp (earliest first)


Conversation-level Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Metadata for each conversation include:

- page_title: The name of the Talk Page where the action occurred.
- page_id: The Wikipedia id of the page on which the action took place.
- page_type: The type of the page (primarily talk vs user talk)
- Example: "287020584.2938.0": {"page_id": "378580", "page_title": "Eric Harris and Dylan Klebold", "page_type": "talk"}


Examples
--------
`Jupyter Notebook containing examples of talk pages with their representations in Wikiconv form <https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/dataset-examples/wikiconv/Create_Conversations_Script.ipynb>`_
