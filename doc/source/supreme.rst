Supreme Court Corpus
====================

A collection of conversations from the U.S. Supreme Court Oral Arguments (51,498 utterances, from 204 cases).

Distributed together with: `Echoes of power: Language effects and power differences in social interaction <https://www.cs.cornell.edu/~cristian/Echoes_of_power.html>`_. Cristian Danescu-Niculescu-Mizil, Bo Pang, Lillian Lee and Jon Kleinberg. WWW 2012


Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

For each speaker, additional information include:

* is-justice: whether the speaker is a Justice
* gender: gender of the speaker


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each utterance, we provide:

* id: index of the utterance
* speaker: the speaker who author the utterance
* root: index of the conversation root of the utterance
* reply_to: index of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance

Metadata for utterances may include:

* case: case number
* justice-is-favorable: true if the Justice eventually vote for this side
* justice-vote: eventual vote from the Justice
* side: side of the case

Note that some utterances may have only a subset of such information.


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("supreme-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 324
Number of Utterances: 51498
Number of Conversations: 938


Additional note
---------------


Related links
^^^^^^^^^^^^^

1. `U.S. Supreme Court Oral Arguments <http://www.supremecourt.gov/oral_arguments/>`_.

2. Case outcome and vote data were extracted from the `Spaeth Supreme Court database <http://scdb.wustl.edu/>`_.

3. This corpus builds upon and enriches the data initially used in: `Computational analysis of the conversational dynamics of the United States Supreme Court <https://drum.lib.umd.edu/handle/1903/9999>`_. Timothy Hawes. Master's Thesis, 2009 and "Elements of a computational model for multi-party discourse: The turn-taking behavior of Supreme Court justices."  Timothy Hawes, Jimmy Lin, and Philip Resnik, JASIST 60(8), 2009.  The original used in these studies can be found `here <https://confluence.cornell.edu/download/attachments/172918779/HAWES_TRANSCRIPT_DATA.zip?version=1&modificationDate=1333554907000&api=v2>`_.

4. A legacy (unmaintained) version of the dataset is `available <https://confluence.cornell.edu/display/llresearch/Supreme+Court+Dialogs+Corpus>`_.

Contact
^^^^^^^
Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil)
