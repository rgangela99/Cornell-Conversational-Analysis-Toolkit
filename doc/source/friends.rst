Friends Corpus
===============

A collection of all the conversations that occurred over 10 seasons of Friends, a popular American TV sitcom that ran in the 1990s.

Across the 10 seasons there are 236 episodes, 3,107 scenes (conversations), 67,373 utterances, and 700 characters (users).

The original dataset is available `here <https://github.com/emorynlp/character-mining>`_. It was originally distributed with `Character Identification on Multiparty Conversation: Identifying Mentions of Characters in TV Shows <https://github.com/emorynlp/character-identification>`_, Henry Y. Chen and Jinho D. Choi. Proceedings of the 17th Annual SIGdial Meeting on Discourse and Dialogue, SIGDIAL'16, 2016.

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset are characters in a given scene. The original dataset provides each character's name as a string, e.g. "Monica Geller". We index Speakers by their names.

Note we add a dummy user named and indexed as "TRANSCRIPT_NOTE" for utterances that are not a character speaking, but instead a note in the transcript, i.e. *"[Time Lapse, Ross has entered.]"*.

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each Utterance we provide:

- id: ``<str>``, the index of the utterance in the format `sAA_eBB_cCC_uDDDD`, where *AA* is the season number, *BB* is the episode number, *CC* is the scene/conversation number, and *DDDD* is the number of the utterance in the scene (e.g. *s01_e18_c05_u021*).
- user: ``<str>``, the user who authored the utterance, aka the speaker, e.g. Monica Geller
- root: ``<str>``, the id of the conversation root of the utterance. We assume conversations begin at the start of a new scene.
- reply_to: ``<str>``, the id of the utterance to which this utterance replies to. `None` if the utterance is the first in a conversation.
- timestamp: ``None``. Our dataset does not contain timestamp information for utterances.
- text: ``<str>``, the textual content of the utterance.

The available metadata varies by seasons, but can include: character entities (or who is referred to in the utterance), emotion, a tokenized version of the text, caption information, and notes about the transcript, which we describe as follows:

- tokens: ``list <str>``, a tokenized representation of the text (useful for sentence separation)
- emotion ``list <str>``, emotion labels for each token. Available for some but not all utterances; `None` if unavailable.
- transcript_with_note: ``<str>``, a version of the text with an action note (e.g. "(to Ross) Hand me the coffee" vs. "Hand me the coffee"). Available for some but not all utterances; ``None`` if unavailable.
- tokens_with_note: ``list <str>``, a tokenized representation of the above.
- caption: ``<str>``, contains the begin time, end time, and text sans punctuation. Available for some but not all utterances; `None` if unavailable. Only available for seasons 6-9.
- character_entities: ``list <str>``, lists of characters who the Speaker is speaking to and/or about. For example, say we have the tokenized utterances *[["There", "'s", "nothing", "to", "tell", "!"],["He", "'s", "just", "some", "guy", "I", "work", "with", "!"]]* and character entities *[[],[[0, 1, "Paul the Wine Guy"], [4, 5, "Paul the Wine Guy"], [5, 6, "Monica Geller"]]]*. The character entities tell us no one gets referred in the first sentence, and in the second sentence, "He" at index 0 and "guy" at index 4 refer to "Paul the Wine Guy", and "I" at index 5 refers to "Monica Geller". Available for some but not all utterances; ``None`` if unavailable.

Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Conversations represent scenes of the show. They are indexed by the id *sXX-eYY-cZZ*, where *XX* denotes the season (e.g. 01), *YY* denotes the episode (e.g. 01), *ZZ* denotes the conversation (e.g. 01).

- season: ``<str>``, the index of the season in the format *sXX*, where XX starts at 01 for season 1 and increments to 10 for season 10.
- episode: ``<str>``, the index of the episode in the format *eXX*, where XX starts at 01 for the first episode of the season and increments accordingly.
- scene: ``<str>``, the index of the scene in the episode in the format *cXX*, where *XX* starts at 01 for the first scene of the episode and increments accordingly. Note that scenes are, for our intents and purposes, conversations.

Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("friends-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 700
Number of Utterances: 67373
Number of Conversations: 3107


Additional note
---------------

Data License
^^^^^^^^^^^^

The original data for this Corpus was obtained from Jinho D. Choi and the Emory NLP team (https://github.com/emorynlp/character-mining). It is copyrighted 2015, Emory University, and licensed under `the Apache License, v.2.0 <https://github.com/emorynlp/character-mining/blob/master/LICENSE.txt>`_.

Contact
^^^^^^^

Please email any questions to Emily Tseng (et397@cornell.edu), Nianyi Wang (nw344@cornell.edu), and Katharine Sadowski (ks2373@cornell.edu).
