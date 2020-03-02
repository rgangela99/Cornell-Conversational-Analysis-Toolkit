# Cornell Conversational Analysis Toolkit ([ConvoKit](http://convokit.cornell.edu/))
This toolkit contains tools to extract conversational features and analyze social phenomena in conversations, using a [single unified interface](https://zissou.infosci.cornell.edu/convokit/documentation/architecture.html) inspired by (and compatible with) scikit-learn.  Several large [conversational datasets](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit#datasets) are included together with scripts exemplifying the use of the toolkit on these datasets. The latest version is [2.3.0](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/releases/tag/v2.3) (released 23 Feb 2020); follow the [project on GitHub](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit) to keep track of updates.

The toolkit currently implements features for:

### [Linguistic coordination](https://www.cs.cornell.edu/~cristian/Echoes_of_power.html) <sub><sup>[(API)](https://zissou.infosci.cornell.edu/convokit/documentation/coordination.html)</sup></sub>

A measure of linguistic influence (and relative power) between individuals or groups based on their use of function words.  
Example: [exploring the balance of power in the U.S. Supreme Court](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/coordination/examples.ipynb).

### [Politeness strategies](https://www.cs.cornell.edu/~cristian/Politeness.html) <sub><sup>[(API)](https://zissou.infosci.cornell.edu/convokit/documentation/politenessStrategies.html)</sup></sub>

A set of lexical and parse-based features correlating with politeness and impoliteness.  
Example: [understanding the (mis)use of politeness strategies in conversations gone awry on Wikipedia](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/conversations-gone-awry/Conversations_Gone_Awry_Prediction.ipynb).

### [Prompt types](http://www.cs.cornell.edu/~cristian/Asking_too_much.html) <sub><sup>[(API)](https://zissou.infosci.cornell.edu/convokit/documentation/promptTypes.html)</sup></sub>

An unsupervised method for grouping utterances and utterance features by their rhetorical role.
Examples: [extracting question types in the U.K. parliament](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/prompt-types/prompt-type-wrapper-demo.ipynb), [extended version demonstrating additional functionality](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/prompt-types/prompt-type-demo.ipynb), [understanding the use of conversational prompts in conversations gone awry on Wikipedia](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/conversations-gone-awry/Conversations_Gone_Awry_Prediction.ipynb).

Also includes functionality to extract surface motifs to represent utterances, used in the above paper [(API)](https://zissou.infosci.cornell.edu/convokit/documentation/phrasingMotifs.html).

### [Hypergraph conversation representation](http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html) <sub><sup>[(API)](https://zissou.infosci.cornell.edu/convokit/documentation/hyperconvo.html)</sup></sub>
A method for extracting structural features of conversations through a hypergraph representation.  
Example: [hypergraph creation and feature extraction, visualization and interpretation on a subsample of Reddit](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/hyperconvo/demo.ipynb).

### [Linguistic diversity in conversations](http://www.cs.cornell.edu/~cristian/Finding_your_voice__linguistic_development.html) <sub><sup>[(API)](https://zissou.infosci.cornell.edu/convokit/documentation/userConvoDiversity.html)</sup></sub>
A method to compute the linguistic diversity of individuals within their own conversations, and between other individuals in a population.  
Example: [user conversation attributes and diversity example on ChangeMyView](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/user-convo-attributes/user-convo-diversity-demo.ipynb)

### [CRAFT: Online forecasting of conversational outcomes](https://arxiv.org/abs/1909.01362) <sub><sup>[(API)](https://zissou.infosci.cornell.edu/convokit/documentation/forecaster.html)</sup></sub>
A neural model for forecasting future outcomes of conversations (e.g., derailment into personal attacks) as they develop.  
Available as an interactive notebook: [fine-tuning](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/convokit/forecaster/CRAFT/demos/craft_demo_training.ipynb) or [inference-only](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/convokit/forecaster/CRAFT/demos/craft_demo_new.ipynb).

## Datasets
ConvoKit ships with several datasets ready for use "out-of-the-box".
These datasets can be downloaded using the `convokit.download()` [helper function](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/convokit/util.py).  Alternatively you can access them directly [here](http://zissou.infosci.cornell.edu/convokit/datasets/).

### [Conversations Gone Awry Dataset](https://zissou.infosci.cornell.edu/convokit/documentation/awry.html)

Two related corpora of conversations that derail into antisocial behavior. One corpus consists of Wikipedia talk page conversations that derail into personal attacks as labeled by crowdworkers (4,188 conversations containing 30.021 comments). The other consists of discussion threads on the subreddit ChangeMyView (CMV) that derail into rule-violating behavior as determined by the presence of a moderator intervention (6,842 conversations containing 42,964 comments).  
Name for download: `conversations-gone-awry-corpus` (Wikipedia version) or `conversations-gone-awry-cmv-corpus` (Reddit CMV version)

### [Cornell Movie-Dialogs Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/movie.html)

A large metadata-rich collection of fictional conversations extracted from raw movie scripts. (220,579 conversational exchanges between 10,292 pairs of movie characters in 617 movies). 
Name for download: `movie-corpus`

### [Parliament Question Time Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/parliament.html)

Parliamentary question periods from May 1979 to December 2016 (216,894 question-answer pairs).  
Name for download: `parliament-corpus`

### [Supreme Court Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/supreme.html)

A collection of conversations from the U.S. Supreme Court Oral Arguments.  
Name for download: `supreme-corpus`

### [Wikipedia Talk Pages Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/wiki.html)

A medium-size collection of conversations from Wikipedia editors' talk pages.  
Name for download: `wiki-corpus`

### [Tennis Interviews](https://zissou.infosci.cornell.edu/convokit/documentation/tennis.html)

Transcripts for tennis singles post-match press conferences for major tournaments between 2007 to 2015 (6,467 post-match press conferences).  
Name for download: `tennis-corpus`

### [Reddit Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/subreddit.html)

Reddit conversations from over 900k subreddits, arranged by subreddit. A [small subset](https://zissou.infosci.cornell.edu/convokit/documentation/reddit-small.html) sampled from 100 highly active subreddits is also available. 
 
Name for download: `subreddit-<name_of_subreddit>` for the by-subreddit data, `reddit-corpus-small` for the small subset. 

### [WikiConv Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/wikiconv.html)

The full corpus of Wikipedia talk page conversations, based on the reconstruction described in [this paper](http://www.cs.cornell.edu/~cristian/index_files/wikiconv-conversation-corpus.pdf).
Note that due to the large size of the data, it is split up by year.
We separately provide [block data retrieved directly from the Wikipedia block log](https://zissou.infosci.cornell.edu/convokit/datasets/wikiconv-corpus/blocks.json), for reproducing the [Trajectories of Blocked Community Members](http://www.cs.cornell.edu/~cristian/Recidivism_online_files/recidivism_online.pdf) paper.

Name for download: `wikiconv-<year>` to download wikiconv data for the specified year.

### [Chromium Conversations Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/chromium.html)

A collection of almost 1.5 million conversations and 2.8 million comments posted by developers reviewing proposed code changes in the Chromium project.

Name for download: `chromium-corpus`

### [Winning Arguments Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/winning.html)

A metadata-rich subset of conversations made in the r/ChangeMyView subreddit between 1 Jan 2013 - 7 May 2015, with information on the delta (success) of a user's utterance in convincing the poster.

Name for download: `winning-args-corpus`

### [Coarse Discourse Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/coarseDiscourse.html)

A subset of Reddit conversations that have been manually annotated with discourse act labels.

Name for download: `reddit-coarse-discourse-corpus`

### [Persuasion For Good Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/persuasionforgood.html)

A collection of online conversations generated by Amazon Mechanical Turk workers, where one participant (the *persuader*) tries to convince the other (the *persuadee*) to donate to a charity.

Name for download: `persuasionforgood-corpus`

### [Intelligence Squared Debates Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/iq2.html)

Transcripts of debates held as part of Intelligence Squared Debates.

Name for download: `iq2-corpus`

### [Friends Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/friends.html)

A collection of all the conversations that occurred over 10 seasons of Friends, a popular American TV sitcom that ran in the 1990s.

Name for download: `friends-corpus`

### [Switchboard Dialog Act Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/switchboard.html)

A collection of 1,155 five-minute telephone conversations between two participants, annotated with speech act tags.

Name for download: `switchboard-corpus`

### ...And your own corpus!

In addition to the provided datasets, you may also use ConvoKit with your own custom datasets by loading them into a `convokit.Corpus` object. [This example script](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/converting_movie_corpus.ipynb) shows how to construct a Corpus from custom data.

## Installation
This toolkit requires Python >= 3.6.

1. Download the toolkit: `pip3 install convokit`
2. Download Spacy's English model: `python3 -m spacy download en`
3. Download NLTK's 'punkt' model: `import nltk; nltk.download('punkt')` (in Python interpreter)

Alternatively, visit our [Github Page](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit) to install from source. 

**If you encounter difficulties with installation**, check out our **[Troubleshooting Guide](https://zissou.infosci.cornell.edu/convokit/documentation/troubleshooting.html)** for a list of solutions to common issues.

## Documentation
Documentation is hosted [here](http://zissou.infosci.cornell.edu/convokit/documentation/). If you are new to ConvoKit, great places to get started are the [Core Concepts tutorial](https://zissou.infosci.cornell.edu/convokit/documentation/architecture.html) for an overview of the ConvoKit "philosophy" and object model, and the [High-level tutorial](https://zissou.infosci.cornell.edu/convokit/documentation/tutorial.html) for an walkthrough of how to import ConvoKit into your project, load a Corpus, and use ConvoKit functions.

## Contributing

We welcome community contributions. To see how you can help out, check the [contribution guidelines](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/CONTRIBUTING.md).

## Citing

If you use the code or datasets distributed with ConvoKit please acknowledge the work tied to the respective component (indicated in the documentation) in addition to:

Jonathan P. Chang, Caleb Chiam, Liye Fu, Andrew Wang, Justine Zhang, Cristian Danescu-Niculescu-Mizil. 2019. "ConvoKit: The Cornell Conversational Analysis Toolkit" Retrieved from http://convokit.cornell.edu

[ConvoKit](http://convokit.cornell.edu/)
