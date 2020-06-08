Deception in Diplomacy Dataset
==============================

Dataset with intended and perceived deception labels in the negotiation-based game Diplomacy, where seven players compete for world domination by forging and breaking alliances with each other.  Over 17,000 messages are annotated by the sender for their intended truthfulness and by the receiver for their perceived truthfulness. This dataset captures deception in long-lasting relationships, where the interlocutors strategically combine truth with lies to advance objectives

Distributed together with:
`It Takes Two to Lie: One to Lie, and One to Listen <https://www.cs.cornell.edu/~cristian/Deception_in_conversations.html>`_. Denis Peskov, Benny Cheng, Ahmed Elgohary, Joe Barrow, Cristian Danescu-Niculescu-Mizil and Jordan Boyd-Graber. Proceedings of ACL 2020.

Dataset details
---------------

The game dynamics and the dataset are described in detail in the paper linked above.

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset are diplomacy players. For each player, we further provide the following information as speaker-level metadata:

For each speaker, we provide:

* id: an unique index of the speaker; in this dataset speakers are distinct between games (even though in reality the same player can participate in multiple games)

Metadata for speakers include:

* country: the country they played in the game


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each utterance, we provide:

* id: index of the utterance
* speaker: the player who authored the utterance
* root: index of the conversation
* reply_to: index of the utterance to which this is a reply to (None if the utterance is not a reply)
* timestamp: the index of the utterance in the game
* text: textual content of the utterance

Metadata for utterances include:

* speaker_intention: “Lie” if the speaker indicated this message was intended to deceive, “Truth” otherwise; this label was provided by the speaker at the time they composed the message.
* receiver_perception: “Lie” if the receiver indicated that they perceived it as deceiving , “Truth” if the receiver indicated that the message was perceived as truthful, None if the receiver did not indicate anything; this label was provided by the receiver at the time they received the message.
* relative_message_index: the index of the utterance in the current conversation
* absolute_message_index: the index of the utterance in the game (same as timestamp)
* year: the Diplomacy-year in which the message was sent
* game_score: the Diplomacy-score the speaker had at the time they sent this message
* game_score_delta: the difference between the Diplomacy-score the speaker and that of the receiver at the time they sent this message
* deception_quadrant: the type of message as defined by both how it was intended and how it was perceived. These are defined as quadrants of Table 3 in the ACL2020 paper: “Straightforward”, “Deceived”, “Caught”, “Cassandra”.


Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A conversation is defined as the totality of exchanged messages between two players in a game.  For each conversation we provide:

* id: an unique index of the conversation;

Metadata for conversations include:

* acl2020_fold: whether the conversation was used in the “Train”, “Test”, or “Validation” set in the ACL2020 paper.



Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download(“diplomacy-corpus"))

For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 83
Number of Utterances: 17289
Number of Conversations: 246

To work only with the Training fold:

>>> corpus_train = Corpus(filename=download(“diplomacy-corpus"))
>>> corpus_train.filter_conversations_by(lambda convo: convo.meta.get('acl2020_fold')=='Train')
>>> corpus_train.print_summary_stats()
Number of Speakers: 62
Number of Utterances: 13132
Number of Conversations: 184



Additional note
---------------

This corpus is also available in a different format at `go.umd.edu/diplomacy_data <https://sites.google.com/view/qanta/projects/diplomacy>`_.

One player was dropped from the data as they did not contribute any messages.

Contact
^^^^^^^

Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil).
