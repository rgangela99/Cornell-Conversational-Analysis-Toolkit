====================
Quick-start tutorial
====================

Setup
=====
Read the `introduction to Convokit <https://convokit.cornell.edu>`_ and the description of its :doc:`architecture </architecture>`.

This toolkit requires Python >=3.6.

If you haven't already,

#. Download the toolkit: ``pip3 install convokit``

#. Download Spacy's English model: ``python3 -m spacy download en``

#. Download nltk's punkt tokenizer: ``import nltk; nltk.download('punkt')`` (in a ``python`` interactive session)

**If you encounter difficulties with installation**, check out our `Troubleshooting Guide <https://zissou.infosci.cornell.edu/convokit/documentation/troubleshooting.html>`_ for a list of solutions to common issues.

Interactive tutorial
====================
Let us start an interactive session (e.g. with ``python`` or ``ipython``) and import Convokit.

>>> import convokit

Now we load an existing corpus, specifically: `reddit-corpus-small`.

By design, it includes 100 comment threads (each consisting of at least 10 Utterances) from 100 popular subreddits from September 2018.

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("reddit-corpus-small"))

Alternatively, if you would like to use a custom corpus, refer to our explanation of the Corpus :doc:`data format </data_format>`.

Exploring the corpus
--------------------

We can examine the corpus metadata:

>>> corpus.meta
{'num_comments': 288846,
 'num_posts': 8286,
 'num_user': 119889,
 'subreddit': 'reddit-corpus-small'}

So the corpus includes 288846 comments and 8286 posts. This is a total of 297132 Utterances. (An Utterance is either a post or a comment in a reddit corpus.)

These 297132 Utterances were made by 119889 different users.

We can get iterators of Utterances, Users, and Conversations, and confirm their sizes match the metadata.

>>> len(list(corpus.iter_users()))
119889
>>> len(list(corpus.iter_utterances()))
297132
>>> len(list(corpus.iter_conversations()))
8286

The iterator functions are the **preferred** way of iterating through Users, Utterances, or Conversations in the Corpus.

We can also get a list of Utterance ids.

>>> utter_ids = corpus.get_utterance_ids()

Let's confirm that there are 297132 Utterances as expected.

>>> len(utter_ids)
297132

Let's take the first Utterance id and examine the Utterance it corresponds to:

>>> utter_ids[0]
'9c716m'
>>> corpus.get_utterance(utter_ids[0])
Utterance({'id': '9c716m', 'user': User([('name', 'AutoModerator')]), 'root': '9c716m', 'reply_to': None, 'timestamp': 1535839576, 'text': 'Talk about your day. Anything goes, but subreddit rules still apply. Please be polite to each other! \n', 'meta': {'score': 13, 'top_level_comment': None, 'retrieved_on': 1540061887, 'gilded': 0, 'gildings': {'gid_1': 0, 'gid_2': 0, 'gid_3': 0}, 'subreddit': 'singapore', 'stickied': False, 'permalink': '/r/singapore/comments/9c716m/rsingapore_random_discussion_and_small_questions/', 'author_flair_text': ''}})

We could also access the first Utterance by using `iter_utterances()`.

>>> next(corpus.iter_utterances())
Utterance({'id': '9c716m', 'user': User([('name', 'AutoModerator')]), 'root': '9c716m', 'reply_to': None, 'timestamp': 1535839576, 'text': 'Talk about your day. Anything goes, but subreddit rules still apply. Please be polite to each other! \n', 'meta': {'score': 13, 'top_level_comment': None, 'retrieved_on': 1540061887, 'gilded': 0, 'gildings': {'gid_1': 0, 'gid_2': 0, 'gid_3': 0}, 'subreddit': 'singapore', 'stickied': False, 'permalink': '/r/singapore/comments/9c716m/rsingapore_random_discussion_and_small_questions/', 'author_flair_text': ''}})

Alternatively, we could access a random Utterance:

>>> corpus.random_utterance()
Utterance({'obj_type': 'utterance', '_owner': <convokit.model.corpus.Corpus object at 0x13adaa410>, 'meta': {'score': 1, 'top_level_comment': 'e5yoyg6', 'retrieved_on': 1539048055, 'gilded': 0, 'gildings': {'gid_1': 0, 'gid_2': 0, 'gid_3': 0}, 'subreddit': 'tifu', 'stickied': False, 'permalink': '/r/tifu/comments/9frsfi/tifu_big_time_i_slept_with_someone_last_night_and/e5yrxtk/', 'author_flair_text': ''}, '_id': 'e5yrxtk', 'user': User({'obj_type': 'user', '_owner': <convokit.model.corpus.Corpus object at 0x13adaa410>, 'meta': {'num_posts': 0, 'num_comments': 2}, '_id': 'condoriano27', '_name': 'condoriano27'}), 'root': '9frsfi', 'reply_to': 'e5ypcii', 'timestamp': 1536933792, 'text': "America's Least Wanted "})

Let's explore the Utterance object further.

>>> utt = next(corpus.iter_utterances())
>>> utt.meta # Utterance-level metadata
{'score': 13,
 'top_level_comment': None,
 'retrieved_on': 1540061887,
 'gilded': 0,
 'gildings': {'gid_1': 0, 'gid_2': 0, 'gid_3': 0},
 'subreddit': 'singapore',
 'stickied': False,
 'permalink': '/r/singapore/comments/9c716m/rsingapore_random_discussion_and_small_questions/',
 'author_flair_text': ''}
>>> utt.id # the identifier for the utterance
'9c716m'
>>> utt.timestamp # the unix timestamp for when the utterance was posted
1535839576
>>> utt.user # the User who posted the Utterance
User([('name', 'AutoModerator')])
>>> utt.user.meta # User-level metadata
{'num_posts': 200, 'num_comments': 27}

Applying a transformer
----------------------

We initialize a Fighting Words transformer, which captures words that capture key differences in speech by two different groups.

For FightingWords specifically, these features are saved to their corresponding Utterance's metadata. Other transformers may update User, Utterance, or Corpus metadata instead.

>>> from convokit import FightingWords
>>> fw = FightingWords()
Initializing default CountVectorizer...

We have to define two groups of utterances between which we would like to find differences in speech:

Let's find the differences between r/atheism and r/Christianity. We define (lambda) filter functions that select for utterances that are in these subreddits.
These functions take an Utterance as input and return True if the Utterance should be included the group.

>>> atheism_only = lambda utt: utt.meta['subreddit'] == 'atheism'
>>> christianity_only = lambda utt: utt.meta['subreddit'] == 'Christianity'

We then pass these filter functions to the ``fit()`` step of Fighting Words in order to train its internal model.

>>> fw.fit(corpus, class1_func=atheism_only, class2_func=christianity_only)
class1_func returned 2736 valid utterances. class2_func returned 2659 valid utterances.
Vocab size is 5059
Comparing language...
ngram zscores computed.

The Fighting Words transformer uses these two functions to define the two classes (groups) of utterances it should compare.
Now that the internal model has been fitted, our Fighting Words transformer has learned which n-grams (i.e. terms) are more important to one group than the other.

We can see a summary of what it has learned using the ``summarize()`` method.

>>> fw.summarize(corpus)
                        z-score   class
ngram
number number        -11.682425  class2
number                -9.647558  class2
god                   -9.557521  class2
sin                   -9.168855  class2
word                  -8.181490  class2
the word              -8.120960  class2
over and              -7.700137  class2
over and over         -7.475561  class2
and over              -7.475561  class2
christ                -7.261349  class2
jesus                 -7.077995  class2
church                -6.887711  class2
gay                   -6.701478  class2
scripture             -6.672350  class2
the church            -6.572954  class2
number number number  -6.142094  class2
homosexuality         -6.112424  class2
of god                -5.946252  class2
bible                 -5.435104  class2
john                  -5.361175  class2
the bible             -5.341622  class2
love                  -5.261977  class2
holy                  -5.243870  class2
men                   -5.010706  class2
israel                -4.994608  class2
god and               -4.935127  class2
with god              -4.829852  class2
heaven                -4.819072  class2
shall                 -4.772242  class2
jewish                -4.753293  class2
...                         ...     ...
atheists               4.369893  class1
government             4.369893  class1
woman                  4.400545  class1
her                    4.401597  class1
atheism                4.574684  class1
circumcision           4.574684  class1
using                  4.583727  class1
human                  4.621385  class1
the article            4.664898  class1
crazy                  4.727097  class1
right to               4.828167  class1
pretty                 4.832246  class1
dont                   4.962440  class1
the woman              4.988421  class1
it                     5.052849  class1
the baby               5.146490  class1
abortion               5.283977  class1
an                     5.318418  class1
fucking                5.464411  class1
story                  5.799971  class1
article                5.804254  class1
shit                   5.806718  class1
url                    5.996616  class1
trump                  6.258077  class1
baby                   6.911191  class1
body                   7.019837  class1
science                7.113479  class1
religious              7.646211  class1
religion               7.817261  class1
money                  7.979943  class1

We get a DataFrame mapping an n-gram to its z-score (a measure of how salient the n-gram is) and the class it belongs to.

As we can see, r/Christianity is comparatively more likely to use terms like 'god', 'sin', and 'christ', while r/atheism uses terms 'money', 'religion', and 'science'.

We also note that there are some (seemingly odd) n-grams like 'number number' and 'url'. This is because FightingWords applies a text cleaner to the Utterance's text prior to model fitting.
This cleaner converts all urls to [url] and numeric values to [number]. (This text cleaning function is configurable.)

This suggests that r/atheism users are more likely to include links in their comments.
As for r/Christianity, their citation of biblical verses, e.g. John 3:16 -> John [number]:[number] -> John number number (after special punctuation is removed), is likely what causes 'number number' to be a salient n-gram.

The Transformer also has other methods for analyzing n-grams now that it is fitted.

>>> # for example, we can check if a given term belongs more in class1 or class2
>>> fw.get_class('state')
'class1'
>>> fw.get_zscore('state')
3.7059870571350846
>>> fw.get_class('spirit')
'class2'
>>> fw.get_zscore('spirit')
-4.136520649529806

Now, we can imagine wanting to annotate Utterances with the fighting words they contain. Say we consider a salient fighting word to be an n-gram with an absolute z-score >= 4.

>>> fw.annot_method = 'threshold' # set the transformer to use the 'threshold' annotation method
>>> fw.threshold = 4 # set threshold value

(Note that these 'annot_method' and 'threshold' parameters can be configured when initializing the Transformer for the first time and are otherwise initialized to default values if not explicitly set.)

>>> # as an example (do not run this)
>>> fw = FightingWords(annot_method='threshold', threshold=4)

Seeing as the corpus contains other subreddits' Utterances that we are not interested in annotating, we can use a selector to get the Transformer to ignore those other Utterances during annotation.

>>> relevant = lambda utt: utt.meta['subreddit'] in ['Christianity', 'atheism']
>>> fw.transform(corpus, selector=relevant)

Let's look at some corpus Utterances from r/Christianity that contain some salient fighting words:

>>> for utt in corpus.iter_utterances(selector=lambda utt: utt.meta['subreddit'] == 'Christianity'):
>>>     print(utt)
>>>     break
Utterance('id': '9c0knf', 'root': 9c0knf, 'reply-to': None, 'user': User('id': Aiming_For_The_Light, 'meta': {'num_posts': 1, 'num_comments': 8}), 'timestamp': 1535778411, 'text': '', 'meta': {'score': 25, 'top_level_comment': None, 'retrieved_on': 1540058137, 'gilded': 0, 'gildings': {'gid_1': 0, 'gid_2': 0, 'gid_3': 0}, 'subreddit': 'Christianity', 'stickied': False, 'permalink': '/r/Christianity/comments/9c0knf/states_expected_to_push_ahead_with_mandatory/', 'author_flair_text': 'Uniting Church in Australia', 'fighting_words_class1': [], 'fighting_words_class2': []})

Notice that meta['fighting_words_class1'] and meta['fighting_words_class1'] are empty lists. This makes sense since this particular Utterance has no text.

Let's refine our selector so that we get what we want:

>>> christianity_salient = lambda utt: utt.meta['subreddit'] == 'Christianity' and len(utt.meta['fighting_words_class2']) > 0
>>> for utt in corpus.iter_utterances(selector=christianity_salient):
>>>     print(utt)
>>>     break
Utterance('id': '9c6un6', 'root': 9c6un6, 'reply-to': None, 'user': User('id': alittlehappy, 'meta': {'num_posts': 1, 'num_comments': 0}), 'timestamp': 1535838106, 'text': "Parents are strict, Orthodox and religious. Father is a priest. I was born in a country where the majority were Orthodox so I've grown up with faith. We moved to American a decade ago and it's been the same since.\n\n\nBut now, I feel so disillusioned. I feel so guilty about this but I simply don't believe in God like I used to. I despise going to church because of how strict it is. My whole family has to get up at 4am and attend church from 5am-10am. Not only that, but we have to stand 95% of the time. Every Sunday, I'm exhausted, bored out of my mind because it's in a language I don't understand and self conscious whenever I sit.\n\n\nI don't know if it's just me losing faith or if I just *really* dislike my church environment. What I wouldn't give to go to a church in the afternoon or late morning with a 2 hour service where I could sit....but I can't even bring it up to my parents because they would 110% take it as a betrayal. I can see why considering my dad preaches/prays in our church so it's like he's not good enough/our religion isn't good enough but ugh.\n\n\nI fear that if they continue to force me and pressure me to go to church I'm going to end up hating Christianity. ", 'meta': {'score': 6, 'top_level_comment': None, 'retrieved_on': 1540061807, 'gilded': 0, 'gildings': {'gid_1': 0, 'gid_2': 0, 'gid_3': 0}, 'subreddit': 'Christianity', 'stickied': False, 'permalink': '/r/Christianity/comments/9c6un6/losing_faith/', 'author_flair_text': '', 'fighting_words_class1': ['religion', 'religious', 'an', 'it', 'her', 'get'], 'fighting_words_class2': ['sin', 'church', 'men', 'and', 'priest', 'am', 'our']})

In summary
----------
We have gone through the application of the Fighting Words to a corpus. Other Transformers follow a similar pattern:

- They are initialized with several configurable parameters.
- They may be ``fit()`` on the Corpus if the Transformer needs to learn something from the Corpus.
- They can ``transform()`` the corpus to annotate its components with the output of the Transformer.
- They can ``summarize()`` their results in a more visual and easily interpreted format -- though in most cases (but not this one), this requires that the Corpus be transformed first.
- These ``fit()``, ``transform()``, ``summarize()`` functions have ``selector`` as an argument so you can further specify subsets of the Corpus to analyze.
- Selectors and filters are typically lambda functions in order to maximize customisability.

Other transformers can be applied in the same way, and even chained in sequence, as described in the :doc:`Core Concepts Tutorial </architecture>`.

Additional notes
----------------

1. Some corpora are particularly large and may not be initializable in their entirety without significant computational resources. However, it is possible to `partially load utterances from a dataset <https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/tests/test_corpus_partial_load.ipynb>`_ to carry out processing of large corpora sequentially.

2. It is possible to `merge two different Corpora (even when there are overlaps or conflicts in Corpus data) <https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/merging/corpus_merge_demo.ipynb>`_

3. See :doc:`examples` for more illustrations of Corpus and Transformer functionality.



