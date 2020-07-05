Oyez Supreme Court Cases Dataset
==============================

A collection of cases from the U.S. Supreme Court, along with transcripts of oral arguments. Extends a `smaller dataset`<https://convokit.cornell.edu/documentation/supreme.html> of oral arguments previously released. 

The corpus is split up into different years spanning 1955 to 2019, each named "oyez_(year)". Additional metadata corresponding to cases, and identities of justices, are also included: `cases`<https://zissou.infosci.cornell.edu/convokit/datasets/oyez-corpus/cases.jsonl> and `justices`<https://zissou.infosci.cornell.edu/convokit/datasets/oyez-corpus/justices.tsv>.

An example of how this corpus is used can be found `here`<https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/tree/master/examples/orientation>.

Note that this corpus is currently in development. The oral arguments should be fairly well-formed and complete, but we are missing information about metadata related to cases: in particular, the decisions of each case and how the justices voted. We note other uncertainties below as well.

Usage
-----

To download a particular year:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("oyez_2019"))



Dataset details
---------------

The data was scraped from the `Oyez`<https://www.oyez.org/> website. Code to scrape and process cases and oral arguments transcripts will be released shortly.


Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers correspond to justices and lawyers (also referred to as advocates). 

For each Speaker, we provide:

* id: the ID of the Speaker. If provided in Oyez, we use this ID, such that further information about advocates or justices may be found at `oyez.org/advocates/<id>` or `oyez.org/justices/<id>`. Otherwise this is inferred (see below)
* name: the name of the Speaker, as listed in transcripts.
* type: whether the speaker is a justice J, advocate A or unknown U.  

Additional details: 

* When possible, we tried to ensure Speaker information corresponds to information provided in Oyez. Oyez usualy provides explicit lists of the speakers involved in each oral argument, especially for more recent cases; earlier ones are missing these explicit lists. Otherwise we tried to follow the Oyez format for converting between names listed in transcripts and IDs (i.e., replacing spaces with underscores and lowercasing).


Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversations correspond to different sessions of oral arguments and re-arguments. Note that a case can have multiple conversations.

For each Conversation, we provide:

* id: we use the ID of the corresponding transcript, as provided by Oyez.
* case_id: the ID of the case (see below).
* advocates: a dictionary where each entry lists the following information for each lawyer:
	* role: the role that the advocate plays (e.g., "Argued for the petitioner"), as listed by Oyez; "inferred" if no role is listed. 
	* side: the side that the advocate is on: 0 for respondent, 1 for petitioner, 2 for amicus curiae (NOTE that we currently do not differentiate between which side the amicus was supporting), 3 for unknown, None for unknown or inaudible speakers (see below, Utterance-level information). If no role is listed in Oyez, this is inferred via some heuristics (documentation forthcoming).
	

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each utterance, we provide:

* id
* text. Oyez seems to separate different sentences into different paragraphs to facilitate its audio-to-text  matching; we've retained this segmentation in the data (a formal check of this is forthcoming). 
* speaker. Note that some utterances have "<INAUDIBLE>" speakers, corresponding to turns listed in the Oyez transcripts without any speaker information, where an interjection was audible but the identity of the speaker couldn't be discerned.
* conversation_id
* case_id: the ID of the case in which the oral argument took place.
* speaker_type: whether the speaker is a justice J, advocate A, or unknown/inaudible U.
* side: the speaker's side (see above, Conversation-level information, and note that this is sometimes inferred from the data if not explicitly listed)
* start_times: the timestamp (as listed in Oyez) of when each sentence in the text starts. There is one entry per sentence, corresponding to newlines in the text.
* stop_times: the timestamp of when each sentence ends.

We also provide dependency parses for each utterance, which can be loaded as:

>>> corpus.load_info('utterance',['parsed'])

Note that at present, each sentence of a parse contains an extra space at the end, due to how Oyez segments different sentences into paragraphs. A todo is to check  that the Oyez segmentation indeed corresponds to sentence breaks (such that the additional newlines can be safely removed).


Justice information
^^^^^^^^^^^^^^^^^^^^^

`This file`<https://zissou.infosci.cornell.edu/convokit/datasets/oyez-corpus/justices.tsv> is a tab-separated table listing the IDs of justices and their names in the transcripts.

Case information
^^^^^^^^^^^^^^^^^^^^^

`This file`<https://zissou.infosci.cornell.edu/convokit/datasets/oyez-corpus/case.jsonl> is a list of json objects containing some information about each case, pulled from Oyez. Note that at present, we don't have information about the decisions made in each case, or the votes of each justice, since such information seems to be inconsistently provided in Oyez. 

* id: generally formatted as <year of case>_<docket no>
* year
* citation: one way to potentially index cases and match with data about decisions
* title: the name of the case
* petitioner: the name of the petitioner
* respondent: the name of the respondent
* docket_no: another way to potentially index cases
* court: the court that saw the case (corresponding to a particular roster of justices)
* url: the url of the Oyez listing
* advocates: the advocates participating in the case. 
* adv_sides_inferred: While most Oyez transcripts explicitly list advocates and their roles, some don't, so we fill this information in via a set of heuristics. This field is True if at least one advocate had information that was filled in in this way.
* transcripts: a list of transcript names, URLs and IDs (corresponding to the IDs of conversations in the corpus). Note that the names almost always contain the date the transcript occurs; we have not presently extracted these dates.


Additional notes
---------------

Code to scrape and process Oyez is forthcoming.

Contact
^^^^^^^

Please email any questions to: jz727@cornell.edu (Justine Zhang).