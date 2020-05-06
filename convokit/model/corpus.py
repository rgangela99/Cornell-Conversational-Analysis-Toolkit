from typing import List, Collection, Callable, Set, Generator, Tuple, Optional, ValuesView, Union
import numpy as np
import pandas as pd
from .corpusHelper import *
from .corpusUtil import warn
from .convoKitIndex import ConvoKitIndex
import random
from .convoKitMeta import ConvoKitMeta

class Corpus:
	"""Represents a dataset, which can be loaded from a folder or a
	list of utterances.
	"""

	def __init__(self, filename: Optional[str] = None, utterances: Optional[List[Utterance]] = None,
				 utterance_start_index: int = None, utterance_end_index: int = None, merge_lines: bool = False,
				 exclude_utterance_meta: Optional[List[str]] = None,
				 exclude_conversation_meta: Optional[List[str]] = None,
				 exclude_user_meta: Optional[List[str]] = None,
				 exclude_overall_meta: Optional[List[str]] = None,
				 version: Optional[int] = None):
		"""

		:param filename: Path to a folder containing a Corpus or to an utterances.jsonl / utterances.json file to load
		:param utterances: List of utterances to initialize Corpus from
		:param utterance_start_index: For utterances.jsonl, specify the line number (zero-indexed) to begin parsing utterances from
		:param utterance_end_index: For utterances.jsonl, specify the line number (zero-indexed) of the last utterance to be parsed.
		:param merge_lines: whether to merge adjacent lines from same user if the two utterances have same root
		:param exclude_utterance_meta: utterance metadata to be ignored
		:param exclude_conversation_meta: conversation metadata to be ignored
		:param exclude_user_meta: user metadata to be ignored
		:param exclude_overall_meta: overall metadata to be ignored
		:param version: version no. of corpus
		"""

		if filename is None:
			self.original_corpus_path = None
		elif os.path.isdir(filename):
			self.original_corpus_path = filename
		else:
			self.original_corpus_path = os.path.dirname(filename)

		self.meta_index = ConvoKitIndex(self)
		self.meta = ConvoKitMeta(self.meta_index, 'corpus')

		self.vector_reprs = {}

		convos_meta = defaultdict(dict)
		if exclude_utterance_meta is None: exclude_utterance_meta = []
		if exclude_conversation_meta is None: exclude_conversation_meta = []
		if exclude_user_meta is None: exclude_user_meta = []
		if exclude_overall_meta is None: exclude_overall_meta = []

		# Construct corpus from file or directory
		if filename is not None:
			if os.path.isdir(filename):
				utterances = load_uttinfo_from_dir(filename, utterance_start_index,
												   utterance_end_index, exclude_utterance_meta)

				users_meta = load_users_meta_from_dir(filename, exclude_user_meta)
				convos_meta = load_convos_meta_from_dir(filename, exclude_conversation_meta)
				load_corpus_meta_from_dir(filename, self.meta, exclude_overall_meta)

				with open(os.path.join(filename, "index.json"), "r") as f:
					idx_dict = json.load(f)
					self.meta_index.update_from_dict(idx_dict)

				# load all processed text information, but don't load actual text.
				# also checks if the index file exists.
				# try:
				#     with open(os.path.join(filename, "processed_text.index.json"), "r") as f:
				#         self.processed_text = {k: {} for k in json.load(f)}
				# except:
				#     pass

				if version is not None:
					if self.meta_index.version != version:
						warn("Requested version does not match file version, "
							 "using file version: {}".format(self.meta_index.version))

				# unpack binary data for utterances
				unpack_binary_data_for_utts(utterances, filename, self.meta_index.utterances_index,
											exclude_utterance_meta, KeyMeta)
				# unpack binary data for users
				unpack_binary_data(filename, users_meta, self.meta_index.users_index, "user",
								   exclude_user_meta)

				# unpack binary data for conversations
				unpack_binary_data(filename, convos_meta, self.meta_index.conversations_index, "convo",
								   exclude_conversation_meta)

				# unpack binary data for overall corpus
				unpack_binary_data(filename, self.meta, self.meta_index.overall_index, "overall", exclude_overall_meta)

			else:
				users_meta = defaultdict(dict)
				convos_meta = defaultdict(dict)
				utterances = load_from_utterance_file(filename, utterance_start_index, utterance_end_index)

			self.utterances = dict()
			self.users = dict()

			initialize_users_and_utterances_objects(self, self.utterances, utterances, self.users, users_meta)

		elif utterances is not None:  # Construct corpus from utterances list
			self.users = {u.user.id: u.user for u in utterances}
			self.utterances = {u.id: u for u in utterances}
			for _, user in self.users.items():
				user.owner = self
			for _, utt in self.utterances.items():
				utt.owner = self

		if merge_lines:
			self.utterances = merge_utterance_lines(self.utterances)

		self.conversations = initialize_conversations(self, self.utterances, convos_meta)
		self.update_users_data()

	@staticmethod
	def dump_helper_bin(d: Dict, d_bin: Dict, object_idx: Dict, fields_to_skip=None) -> Dict:
		"""

		:param d: The dict to encode
		:param d_bin: The dict of accumulated lists of binary attribs
		:param object_idx:
		:return:
		"""
		if fields_to_skip is None:
			fields_to_skip = []

		d_out = {}
		for k, v in d.items():
			if k in fields_to_skip: continue
			try:  # try saving the field
				json.dumps(v)
				d_out[k] = v
				if k not in object_idx:
					object_idx[k] = str(type(v))
			except (TypeError, OverflowError):  # unserializable
				d_out[k] = "{}{}{}".format(BIN_DELIM_L, len(d_bin[k]), BIN_DELIM_R)
				d_bin[k].append(v)
				object_idx[k] = "bin"  # overwrite non-bin type annotation if necessary
		return d_out

	def dump(self, name: str, base_path: Optional[str] = None,
			 increment_version: bool = True,
			 save_to_existing_path: bool = False,
			 fields_to_skip=None) -> None:
		"""
		Dumps the corpus and its metadata to disk. Automatically increments the version number.

		:param name: name of corpus
		:param base_path: base directory to save corpus in (None to save to a default directory)
		:param increment_version: whether to increment the Corpus version number when dumping
		:param save_to_existing_path: if True, save to the path you loaded the corpus from (supersedes base_path)
		:param fields_to_skip: a dictionary of {object type: list of attributes to omit when writing to disk}. object types can be one of "user", "utterance", "conversation", "corpus".
		"""
		if fields_to_skip is None:
			fields_to_skip = dict()
		dir_name = name
		if base_path is not None and save_to_existing_path:
			raise ValueError("Not allowed to specify both base_path and save_to_existing_path!")
		if save_to_existing_path and self.original_corpus_path is None:
			raise ValueError("Cannot use save to existing path on Corpus generated from utterance list!")
		if not save_to_existing_path:
			if base_path is None:
				base_path = os.path.expanduser("~/.convokit/")
				if not os.path.exists(base_path):
					os.mkdir(base_path)
				base_path = os.path.join(base_path, "saved-corpora/")
				if not os.path.exists(base_path):
					os.mkdir(base_path)
			dir_name = os.path.join(base_path, dir_name)
		else:
			dir_name = os.path.join(self.original_corpus_path, name)

		if not os.path.exists(dir_name):
			os.mkdir(dir_name)

		utterances_idx, users_idx, convos_idx, overall_idx = {}, {}, {}, {}  # TODO get rid of this

		dump_corpus_object(self, dir_name, "users.json", "user", "user", fields_to_skip)
		dump_corpus_object(self, dir_name, "conversations.json", "conversation", "convo", fields_to_skip)
		dump_utterances(self, dir_name, fields_to_skip)

		with open(os.path.join(dir_name, "corpus.json"), "w") as f:
			d_bin = defaultdict(list)
			meta_up = Corpus.dump_helper_bin(self.meta, d_bin, overall_idx, fields_to_skip.get('corpus', []))
			#            keys = ["utterances-index", "conversations-index", "users-index",
			#                "overall-index"]
			#            meta_minus = {k: v for k, v in overall_idx.items() if k not in keys}
			#            meta_up["overall-index"] = meta_minus
			json.dump(meta_up, f)
			for name, l_bin in d_bin.items():
				with open(os.path.join(dir_name, name + "-overall-bin.p"), "wb") as f_pk:
					pickle.dump(l_bin, f_pk)

		with open(os.path.join(dir_name, "index.json"), "w") as f:
			json.dump(self.meta_index.to_dict(increment_version=increment_version), f)

	# with open(os.path.join(dir_name, "processed_text.index.json"), "w") as f:
	#     json.dump(list(self.processed_text.keys()), f)

	def get_utterance(self, utt_id: str) -> Utterance:
		"""
		Gets Utterance of the specified id from the corpus

		:param utt_id: id of Utterance
		:return: Utterance
		"""
		return self.utterances[utt_id]

	def get_conversation(self, convo_id: str) -> Conversation:
		"""
		Gets Conversation of the specified id from the corpus

		:param convo_id: id of Conversation
		:return: Conversation
		"""
		return self.conversations[convo_id]

	def get_user(self, user_id: str) -> User:
		"""
		Gets User of the specified id from the corpus

		:param user_id: id of User
		:return: User
		"""
		return self.users[user_id]

	def get_object(self, obj_type: str, oid: str):
		"""
		General Corpus object getter. Gets User / Utterance / Conversation of specified id from the Corpus

		:param obj_type: "user", "utterance", or "conversation"
		:param oid: object id
		:return: Corpus object of specified object type with specified object id
		"""
		assert obj_type in ["user", "utterance", "conversation"]
		if obj_type == "user":
			return self.get_user(oid)
		elif obj_type == "utterance":
			return self.get_utterance(oid)
		else:
			return self.get_conversation(oid)

	def has_utterance(self, utt_id: str) -> bool:
		"""
		Checks if an Utterance of the specified id exists in the Corpus

		:param utt_id: id of Utterance
		:return: True if Utterance of specified id is present, False otherwise
		"""
		return utt_id in self.utterances

	def has_conversation(self, convo_id: str) -> bool:
		"""
		Checks if a Conversation of the specified id exists in the Corpus

		:param convo_id: id of Conversation
		:return: True if Conversation of specified id is present, False otherwise
		"""
		return convo_id in self.conversations

	def has_user(self, user_id: str) -> bool:
		"""
		Checks if a User of the specified id exists in the Corpus

		:param user_id: id of User
		:return: True if User of specified id is present, False otherwise
		"""
		return user_id in self.users

	def random_utterance(self) -> Utterance:
		"""
		Get a random Utterance from the Corpus

		:return: a random Utterance
		"""
		return random.choice(list(self.utterances.values()))

	def random_conversation(self) -> Conversation:
		"""
		Get a random Conversation from the Corpus

		:return: a random Conversation
		"""
		return random.choice(list(self.conversations.values()))

	def random_user(self) -> User:
		"""
		Get a random User from the Corpus

		:return: a random User
		"""
		return random.choice(list(self.users.values()))

	def iter_utterances(self, selector: Optional[Callable[[Utterance], bool]] = lambda utt: True) -> Generator[
		Utterance, None, None]:
		"""
		Get utterances in the Corpus, with an optional selector that filters for Utterances that should be included

		:param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude).
		By default, the selector includes all Utterances in the Corpus.
		:return: a generator of Utterances
		"""
		for v in self.utterances.values():
			if selector(v):
				yield v

	def view_utterances(self, selector: Optional[Callable[[Utterance], bool]] = lambda utt: True):
		"""
		Get utterances in the Corpus, with an optional selector that filters for Utterances that should be included
		Returns a dataframe of the utterances with data and metadata values
		:param selector:
		:return:
		"""
		for utt in self.iter_utterances(selector):
			pass

	def iter_conversations(self, selector: Optional[Callable[[Conversation], bool]] = lambda convo: True) -> Generator[
		Conversation, None, None]:
		"""
		Get conversations in the Corpus, with an optional selector that filters for Conversations that should be included

		:param selector: a (lambda) function that takes a Conversation and returns True or False (i.e. include / exclude).
		By default, the selector includes all Conversations in the Corpus.
		:return: a generator of Conversations
		"""
		for v in self.conversations.values():
			if selector(v):
				yield v

	def iter_users(self, selector: Optional[Callable[[User], bool]] = lambda user: True) -> Generator[User, None, None]:
		"""
		Get Users in the Corpus, with an optional selector that filters for Conversations that should be included

		:param selector: a (lambda) function that takes a User and returns True or False (i.e. include / exclude).
		By default, the selector includes all Users in the Corpus.
		:return: a generator of Users
		"""

		for user in self.users.values():
			if selector(user):
				yield user

	def iter_objs(self, obj_type: str, selector: Callable[[Union[User, Utterance, Conversation]], bool] = lambda obj: True):
		"""
		Get Corpus objects of specified type from the Corpus, with an optional selector that filters for Corpus object that should be included

		:param obj_type: "user", "utterance", or "conversation"
		:param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude).
		By default, the selector includes all objects of the specified type in the Corpus.
		:return: a generator of Users
		"""

		assert obj_type in ["user", "utterance", "conversation"]
		obj_iters = {"conversation": self.iter_conversations,
					 "user": self.iter_users,
					 "utterance": self.iter_utterances}

		return obj_iters[obj_type](selector)

	def get_utterance_ids(self, selector: Optional[Callable[[Utterance], bool]] = lambda utt: True) -> List[str]:
		"""
		Get a list of ids of Utterances in the Corpus, with an optional selector that filters for Utterances that should be included

		:param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude).
		By default, the selector includes all Utterances in the Corpus.
		:return: list of Utterance ids
		"""
		return [utt.id for utt in self.iter_utterances(selector)]

	def get_conversation_ids(self, selector: Optional[Callable[[Conversation], bool]] = lambda convo: True) -> List[
		str]:
		"""
		Get a list of ids of Conversations in the Corpus, with an optional selector that filters for Conversations that should be included

		:param selector: a (lambda) function that takes a Conversation and returns True or False (i.e. include / exclude).
		By default, the selector includes all Conversations in the Corpus.
		:return: list of Conversation ids
		"""
		return [convo.id for convo in self.iter_conversations(selector)]

	def get_user_ids(self, selector: Optional[Callable[[User], bool]] = lambda user: True) -> List[
		str]:
		"""
		Get a list of ids of Users in the Corpus, with an optional selector that filters for Users that should be included

		:param selector: a (lambda) function that takes a User and returns True or False (i.e. include / exclude).
		By default, the selector includes all Users in the Corpus.
		:return: list of User ids
		"""
		return [user.id for user in self.iter_users(selector)]

	def get_object_ids(self, obj_type: str,
					   selector: Callable[[Union[User, Utterance, Conversation]], bool] = lambda obj: True):
		"""
		Get a list of ids of Corpus objects of the specified type in the Corpus, with an optional selector that filters for objects that should be included

		:param obj_type: "user", "utterance", or "conversation"
		:param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude).
		By default, the selector includes all objects of the specified type in the Corpus.
		:return: list of Corpus object ids
		"""
		assert obj_type in ["user", "utterance", "conversation"]
		return [obj.id for obj in self.iter_objs(obj_type, selector)]

	def get_usernames(self, selector: Optional[Callable[[User], bool]] = lambda user: True) -> Set[str]:
		"""Get names of users in the dataset.

		This function will be deprecated and replaced by get_user_ids()

		:param selector: optional function that takes in a
			`User` and returns True to include the user's name in the
			resulting list, or False otherwise.

		:return: Set containing all user names selected by the selector
			function, or all user names in the dataset if no selector function
			was used.

		"""
		warn("This function is deprecated. Use get_user_ids() instead.")
		return set([u.id for u in self.iter_users(selector)])

	def filter_conversations_by(self, selector: Callable[[Conversation], bool]):
		"""
		Mutate the corpus by filtering for a subset of Conversations within the Corpus

		:param selector: function for selecting which functions to keep
		:return: None (mutates the corpus)
		"""

		self.conversations = {convo_id: convo for convo_id, convo in self.conversations.items() if selector(convo)}
		utt_ids = set([utt for convo in self.conversations.values() for utt in convo.get_utterance_ids()])
		self.utterances = {utt.id: utt for utt in self.utterances.values() if utt.id in utt_ids}
		usernames = set([utt.user.id for utt in self.utterances.values()])
		self.users = {user.id: user for user in self.users.values() if user.id in usernames}
		self.update_users_data()
		self.reinitialize_index()

	def reindex_conversations(self, new_convo_roots: List[str], preserve_corpus_meta: bool = True,
							  preserve_convo_meta: bool = True, verbose = True) -> 'Corpus':
		"""
		Generates a new Corpus from current Corpus with specified list of utterance ids to use as conversation roots.

		The subtrees denoted by these utterance ids should be distinct and should not overlap, otherwise there may be unexpected behavior.

		:param new_convo_roots: List of utterance ids to use as roots
		:param preserve_corpus_meta: set as True to copy original Corpus metadata to new Corpus
		:param preserve_convo_meta: set as True to copy original Conversation metadata to new Conversation metadata
		(For each new convo root, use the metadata of the conversation that convo root belonged to.)
		:param verbose: whether to print a warning when
		:return: new Corpus with reindexed Conversations
		"""""
		new_convo_roots = set(new_convo_roots)
		for convo in self.iter_conversations():
			try:
				convo.initialize_tree_structure()
			except ValueError as e:
				if verbose:
					warn(str(e))

		new_corpus_utts = []
		original_utt_to_convo_id = dict()

		for utt_id in new_convo_roots:
			orig_convo = self.get_conversation(self.get_utterance(utt_id).root)
			original_utt_to_convo_id[utt_id] = orig_convo.id
			try:
				subtree = orig_convo.get_subtree(utt_id)
				new_root_utt = subtree.utt
				new_root_utt.reply_to = None
				subtree_utts = [node.utt for node in subtree.bfs_traversal()]
				for utt in subtree_utts:
					utt.root = utt_id
				new_corpus_utts.extend(subtree_utts)
			except ValueError:
				continue

		new_corpus = Corpus(utterances=new_corpus_utts)

		if preserve_corpus_meta:
			new_corpus.meta.update(self.meta)

		if preserve_convo_meta:
			for convo in new_corpus.iter_conversations():
				convo.meta['original_convo_meta'] = self.get_conversation(original_utt_to_convo_id[convo.id]).meta
				convo.meta['original_convo_id'] = original_utt_to_convo_id[convo.id]
		if verbose:
			missing_convo_roots = list(set(new_convo_roots) - set(new_corpus.get_conversation_ids()))
			if len(missing_convo_roots) > 0:
				warn("Failed to find some of the specified new convo roots:\n")
				print(missing_convo_roots)

		return new_corpus

	def utterance_threads(self, prefix_len: Optional[int] = None,
						  suffix_len: int = 0,
						  include_root: bool = True) -> Dict[str, Dict[str, Utterance]]:
		"""
		Returns dict of threads, where a thread is all utterances with the
		same root.

		:param prefix_len: if an integer n, only get the first n utterances
			of each thread (sorted by ascending timestamp value)
		:param suffix_len: if an integer n, only get the last n utterances
			of each thread (sorted by descending timestamp value)
		:param include_root: True if root utterance should be included in the utterance thread.
			If False, thread begins from top level comment.

		:return: Dictionary from thread root ids to threads, where a thread is
			itself a dictionary from utterance ids to utterances.
		"""
		threads = defaultdict(list)
		for ut in self.utterances.values():
			if include_root:
				threads[ut.root].append(ut)
			else:
				top_level_comment = ut.meta["top_level_comment"]
				if top_level_comment is None: continue  # i.e. this is a post (root) utterance
				threads[top_level_comment].append(ut)
		return {root: {utt.id: utt for utt in list(sorted(l, key=lambda x: x.timestamp))[-suffix_len:prefix_len]}
				for root, l in threads.items()}

	def get_meta(self) -> Dict:
		return self.meta

	def add_meta(self, key: str, value) -> None:
		self.meta[key] = value



	def speaking_pairs(self, selector: Optional[Callable[[User, User], bool]] = lambda user1, user2: True,
					   user_names_only: bool = False) -> Set[Tuple[str, str]]:
		"""Get all directed speaking pairs (a, b) of users such that a replies
			to b at least once in the dataset.

		:param selector: optional function that takes in
			a speaker user and a replied-to user and returns True to include
			the pair in the result, or False otherwise.
		:param user_names_only: if True, return just pairs of
			user names rather than user objects.
		:type user_names_only: bool

		:return: Set containing all speaking pairs selected by the selector
			function, or all speaking pairs in the dataset if no selector
			function was used.
		"""
		pairs = set()
		for utt2 in self.iter_utterances():
			if utt2.user is not None and utt2.reply_to is not None and utt2.reply_to in self.utterances:
				utt1 = self.get_utterance(utt2.reply_to)
				if utt1.user is not None:
					if selector(utt2.user, utt1.user):
						pairs.add((utt2.user.id, utt1.user.id) if
								  user_names_only else (utt2.user, utt1.user))
		return pairs

	def pairwise_exchanges(self, selector: Optional[Callable[[User, User], bool]] = None,
						   user_names_only: bool = False) -> Dict[Tuple, List[Utterance]]:
		"""
		Get all directed pairwise exchanges in the dataset.

		:param selector: optional function that takes in a
			speaker user and a replied-to user and returns True to include
			the pair in the result, or False otherwise.
		:param user_names_only: if True, index conversations
			by user names rather than user objects.
		:type user_names_only: bool

		:return: Dictionary mapping (speaker, target) tuples to a list of
			utterances given by the speaker in reply to the target.
		"""
		pairs = defaultdict(list)
		for u2 in self.utterances.values():
			if u2.user is not None and u2.reply_to is not None:
				u1 = self.utterances[u2.reply_to]
				if u1.user is not None:
					if selector is None or selector(u2.user, u1.user):
						key = (u2.user.id, u1.user.id) if user_names_only else (u2.user, u1.user)
						pairs[key].append(u2)
		return pairs

	def iterate_by(self, iter_type: str,
				   is_utterance_question: Callable[[str], bool]) -> Generator[Tuple[str, str, str], None, None]:
		"""
		Iterator for utterances.

		Can give just questions, just answers or questions followed by their answers
		"""

		pair_delim = '-q-a-'
		i = -1
		for utterance in self.utterances.values():
			if utterance.reply_to is not None:
				root_text = self.utterances[utterance.reply_to].text
				if is_utterance_question(root_text):
					i += 1
					if iter_type == 'answers':
						pair_idx = utterance.reply_to + pair_delim + str(utterance.id)
						yield utterance.id, utterance.text, pair_idx
						continue
					question = self.utterances[utterance.reply_to]
					pair_idx = str(question.id) + pair_delim + str(utterance.id)
					yield question.id, question.text, pair_idx
					if iter_type == 'both':
						pair_idx = utterance.reply_to + pair_delim + str(utterance.id)
						yield utterance.id, utterance.text, pair_idx

	@staticmethod
	def _merge_utterances(utts1: List[Utterance], utts2: List[Utterance], warnings: bool) -> ValuesView[Utterance]:
		"""
		Helper function for merge().

		Combine two collections of utterances into a single dictionary of Utterance id -> Utterance.

		If metadata of utterances in the two collections share the same key, but different values,
		the second collections' utterance metadata will be used.

		May mutate original collections of Utterances.

		Prints warnings when:
		1) Utterances with same id from this and other collection do not share the same data
		2) Utterance metadata has different values for the same key, and overwriting occurs

		:param utts1: First collection of Utterances
		:param utts2: Second collection of Utterances
		:param warnings: whether to print warnings when conflicting data is found.
		:return: ValuesView for merged set of utterances
		"""
		seen_utts = dict()

		# Merge UTTERANCE metadata
		# Add all the utterances from this corpus
		for utt in utts1:
			seen_utts[utt.id] = utt

		# Add all the utterances from the other corpus, checking for data sameness and updating metadata as appropriate
		for utt in utts2:
			if utt.id in seen_utts:
				prev_utt = seen_utts[utt.id]
				if prev_utt == utt:
					# other utterance metadata is ignored if data is not matched
					for key, val in utt.meta.items():
						if key in prev_utt.meta and prev_utt.meta[key] != val:
							if warnings:
								warn("Found conflicting values for Utterance {} for metadata key: {}. "
									 "Overwriting with other corpus's Utterance metadata.".format(repr(utt.id),
																								  repr(key)))
						prev_utt.meta[key] = val
				else:
					if warnings:
						warn("Utterances with same id do not share the same data:\n" +
							 str(prev_utt) + "\n" +
							 str(utt) + "\n" +
							 "Ignoring second corpus's utterance."
							 )
			else:
				seen_utts[utt.id] = utt

		return seen_utts.values()

	@staticmethod
	def _collect_user_data(utt_sets: Collection[Collection[Utterance]]) -> Tuple[
		Dict[str, Dict[str, str]], Dict[str, Dict[str, bool]]]:
		"""
		Helper function for merge().

		Iterates through the input set of utterances, to collect User data and metadata.

		Collect User metadata in another Dictionary indexed by User ID

		Track if conflicting user metadata is found in another dictionary

		:param utt_sets: Collections of collections of Utterances to extract Users from
		:return: user metadata and the corresponding tracker
		"""
		# Collect USER data and metadata
		# users_data = defaultdict(lambda: defaultdict(set))
		users_meta = defaultdict(lambda: defaultdict(str))
		users_meta_conflict = defaultdict(lambda: defaultdict(bool))
		for utt_set in utt_sets:
			for utt in utt_set:
				for meta_key, meta_val in utt.user.meta.items():
					curr = users_meta[utt.user][meta_key]
					if curr != meta_val:
						if curr != "":
							users_meta_conflict[utt.user][meta_key] = True
						users_meta[utt.user][meta_key] = meta_val

		return users_meta, users_meta_conflict

	@staticmethod
	def _update_corpus_user_data(new_corpus, users_meta: Dict, users_meta_conflict: Dict, warnings: bool) -> None:
		"""
		Helper function for merge().

		Update new_corpus's Users' data (utterance and conversation lists) and metadata

		Prints a warning if multiple values are found for any user's metadata key; latest user metadata is used

		:param users_meta: Dictionary indexed by User ID, containing the collected User metadata
		:param users_meta_conflict: Dictionary indexed by User ID, indicating if there were value conflicts for the associated meta keys
		:return: None (mutates the new_corpus's Users)
		"""
		# Update USER data and metadata with merged versions
		for user in new_corpus.iter_users():
			for meta_key, meta_val in users_meta[user].items():
				if users_meta_conflict[user][meta_key]:
					if warnings:
						warn("Multiple values found for {} for metadata key: {}. "
							 "Taking the latest one found".format(user, repr(meta_key)))
				user.meta[meta_key] = meta_val

	def _reinitialize_index_helper(self, new_index, old_index, obj_type):
		"""
		Helper for reinitializing the index of the different Corpus object types
		:param new_index: new ConvoKitIndex object
		:param old_index: original ConvoKitIndex object
		:param obj_type: utterance, user, or conversation
		:return: None (mutates new_index)
		"""
		new_obj_index = new_index.indices[obj_type]
		old_obj_index = old_index.indices[obj_type]

		for obj in self.iter_objs(obj_type):
			for key, value in obj.meta.items():
				if key in new_obj_index:
					if new_obj_index[key] is None and value is not None:
						new_obj_index[key] = str(type(value))
				else:
					if key in old_obj_index:
						new_obj_index[key] = old_obj_index[key]
					else:
						new_obj_index[key] = str(type(value))

	def reinitialize_index(self):
		"""
		Reinitialize the Corpus Index. Called during merge().
		Re-uses original Index values where possible, and avoids having NoneType as the class-type for any key.
		Checks metadata of all Corpus objects of each type to ensure that all keys are accounted for.

		:return: None (sets the .meta_index of Corpus)
		"""
		old_index = self.meta_index
		new_index = ConvoKitIndex(self)

		self._reinitialize_index_helper(new_index, old_index, "utterance")
		self._reinitialize_index_helper(new_index, old_index, "user")
		self._reinitialize_index_helper(new_index, old_index, "conversation")

		for key, value in self.meta.items():  # overall
			new_index.overall_index[key] = str(type(value))

		new_index.version = old_index.version
		self.meta_index = new_index

	def merge(self, other_corpus, warnings: bool = True):
		"""
		Merges this corpus with another corpus.

		Utterances with the same id must share the same data, otherwise the other corpus utterance data & metadata
		will be ignored. A warning is printed when this happens.

		If metadata of this corpus (or its conversations / utterances) shares a key with the metadata of the
		other corpus, the other corpus's metadata (or its conversations / utterances) values will be used. A warning
		is printed when this happens.

		May mutate original and other corpus in the process.

		(Updates internal ConvoKit Index to match post-merge state and uses this Corpus's version number.)

		:param other_corpus: Corpus
		:param warnings: print warnings when data conflicts are encountered
		:return: new Corpus constructed from combined lists of utterances
		"""
		utts1 = list(self.iter_utterances())
		utts2 = list(other_corpus.iter_utterances())
		combined_utts = self._merge_utterances(utts1, utts2, warnings=warnings)
		new_corpus = Corpus(utterances=list(combined_utts))
		# Note that we collect Users from the utt sets directly instead of the combined utts, otherwise
		# differences in User meta will not be registered for duplicate Utterances (because utts would be discarded
		# during merging)
		users_meta, users_meta_conflict = self._collect_user_data([utts1, utts2])
		Corpus._update_corpus_user_data(new_corpus, users_meta, users_meta_conflict, warnings=warnings)

		# Merge CORPUS metadata
		new_corpus.meta = self.meta
		for key, val in other_corpus.meta.items():
			if key in new_corpus.meta and new_corpus.meta[key] != val:
				if warnings:
					warn("Found conflicting values for Corpus metadata key: {}. "
						 "Overwriting with other Corpus's metadata.".format(repr(key)))
			new_corpus.meta[key] = val

		# Merge CONVERSATION metadata
		convos1 = self.iter_conversations()
		convos2 = other_corpus.iter_conversations()

		for convo in convos1:
			new_corpus.get_conversation(convo.id).meta = convo.meta

		for convo in convos2:
			for key, val in convo.meta.items():
				curr_meta = new_corpus.get_conversation(convo.id).meta
				if key in curr_meta and curr_meta[key] != val:
					if warnings:
						warn("Found conflicting values for Conversation {} for metadata key: {}. "
							 "Overwriting with other corpus's Conversation metadata.".format(repr(convo.id), repr(key)))
				curr_meta[key] = val

		new_corpus.update_users_data()
		new_corpus.reinitialize_index()

		return new_corpus

	def add_utterances(self, utterances=List[Utterance], warnings: bool = False, with_checks=True):
		"""
		Add utterances to the Corpus.

		If the corpus has utterances that share an id with an utterance in the input utterance list,

		Optional warnings will be printed:
		- if the utterances with same id do not share the same data (added utterance is ignored)
		- added utterances' metadata have the same key but different values (added utterance's metadata will overwrite)

		:param utterances: Utterances to be added to the Corpus
		:param warnings: set to True for warnings to be printed
		:param with_checks: set to True if checks on utterance and metadata overlaps are desired. Set to False if newly added utterances are guaranteed to be new and share the same set of metadata keys.
		:return: a Corpus with the utterances from this Corpus and the input utterances combined
		"""
		if with_checks:
			helper_corpus = Corpus(utterances=utterances)
			return self.merge(helper_corpus, warnings=warnings)
		else:
			new_users = {u.user.id: u.user for u in utterances}
			new_utterances = {u.id: u for u in utterances}
			for user in new_users.values():
				user.owner = self
			for utt in new_utterances.values():
				utt.owner = self

			# update corpus users
			for new_user_id, new_user in new_users.items():
				if new_user_id not in self.users:
					self.users[new_user_id] = new_user

			# update corpus utterances + (link user -> utt)
			for new_utt_id, new_utt in new_utterances.items():
				if new_utt_id not in self.utterances:
					self.utterances[new_utt_id] = new_utt
					self.users[new_utt.user.id]._add_utterance(new_utt)

			# update corpus conversations + (link convo <-> utt)
			new_convos = defaultdict(list)
			for utt in new_utterances.values():
				if utt.root in self.conversations:
					self.conversations[utt.root]._add_utterance(utt)
				else:
					new_convos[utt.root].append(utt.id)
			for convo_id, convo_utts in new_convos.items():
				new_convo = Conversation(owner=self, id=convo_id,
										 utterances=convo_utts,
										 meta=None)
				self.conversations[convo_id] = new_convo
				# (link user -> convo)
				new_convo_user = self.users[new_convo.get_utterance(convo_id).user.id]
				new_convo_user._add_conversation(new_convo)
		return self

	def update_users_data(self) -> None:
		"""
		Updates the conversation and utterance lists of every User in the Corpus

		:return: None
		"""
		users_utts = defaultdict(list)
		users_convos = defaultdict(list)

		for utt in self.iter_utterances():
			users_utts[utt.user.id].append(utt)

		for convo in self.iter_conversations():
			for utt in convo.iter_utterances():
				users_convos[utt.user.id].append(convo)

		for user in self.iter_users():
			user.utterances = {utt.id: utt for utt in users_utts[user.id]}
			user.conversations = {convo.id: convo for convo in users_convos[user.id]}

	def print_summary_stats(self) -> None:
		"""
		Helper function for printing the number of Users, Utterances, and Conversations in this Corpus

		:return: None
		"""
		print("Number of Users: {}".format(len(self.users)))
		print("Number of Utterances: {}".format(len(self.utterances)))
		print("Number of Conversations: {}".format(len(self.conversations)))

	def get_vect_repr(self, id, field):
		"""
		gets a vector representation stored under the name field, for an individual object with a particular id.

		:param id: id of object
		:param field: the name of the particular representation
		:return: a vector representation of object <id>
		"""

		vect_obj = self.vector_reprs[field]
		try:
			idx = vect_obj['key_to_idx'][id]
			return vect_obj['vects'][idx]
		except KeyError:
			return None

	def set_vect_reprs(self, field, keys, vects):
		"""
		stores a matrix where each row is a vector representation of an object

		:param field: name of representation
		:param keys: list of object ids, where each entry corresponds to each row of the matrix
		:param vects: matrix of vector representations
		:return: None
		"""

		vect_obj = {'vects': vects, 'keys': keys}
		vect_obj['key_to_idx'] = {k: idx for idx, k in enumerate(vect_obj['keys'])}
		self.vector_reprs[field] = vect_obj

	@staticmethod
	def _load_jsonlist_to_dict(filename, index_key='id', value_key='value'):
		entries = {}
		with open(filename, 'r') as f:
			for line in f:
				entry = json.loads(line)
				entries[entry[index_key]] = entry[value_key]
		return entries

	@staticmethod
	def _dump_jsonlist_from_dict(entries, filename, index_key='id', value_key='value'):
		with open(filename, 'w') as f:
			for k, v in entries.items():
				json.dump({index_key: k, value_key: v}, f)
				f.write('\n')

	@staticmethod
	def _load_vectors(filename):
		vect_obj = {}
		with open(filename + '.keys') as f:
			vect_obj['keys'] = [x.strip() for x in f.readlines()]
		vect_obj['key_to_idx'] = {k: idx for idx, k in enumerate(vect_obj['keys'])}
		vect_obj['vects'] = np.load(filename + '.npy')
		return vect_obj

	@staticmethod
	def _dump_vectors(vect_obj, filename):
		with open(filename + '.keys', 'w') as f:
			f.write('\n'.join(vect_obj['keys']))
		np.save(filename, vect_obj['vects'])

	def load_info(self, obj_type, fields=None, dir_name=None):
		"""
		loads attributes of objects in a corpus from disk.
		This function, along with dump_info, supports cases where a particular attribute is to be stored separately from the other corpus files, for organization or efficiency. These attributes will not be read when the corpus is initialized; rather, they can be loaded on-demand using this function.

		For each attribute with name <NAME>, will read from a file called info.<NAME>.jsonl, and load each attribute value into the respective object's .meta field.

		:param obj_type: type of object the attribute is associated with. can be one of "utterance", "user", "conversation".
		:param fields: a list of names of attributes to load. if empty, will load all attributes stored in the specified directory dir_name.
		:param dir_name: the directory to read attributes from. by default, or if set to None, will read from the directory that the Corpus was loaded from.
		:return: None
		"""
		if fields is None:
			fields = []

		if (self.original_corpus_path is None) and (dir_name is None):
			raise ValueError('must specify a directory to read from')
		if dir_name is None:
			dir_name = self.original_corpus_path

		if len(fields) == 0:
			fields = [x.replace('info.', '').replace('.jsonl', '') for x in os.listdir(dir_name)
					  if x.startswith('info')]

		for field in fields:
			# self.aux_info[field] = self._load_jsonlist_to_dict(
			#     os.path.join(dir_name, 'feat.%s.jsonl' % field))
			getter = lambda oid: self.get_object(obj_type, oid)
			entries = self._load_jsonlist_to_dict(
				os.path.join(dir_name, 'info.%s.jsonl' % field))
			for k, v in entries.items():
				try:
					obj = getter(k)
					obj.set_info(field, v)
				except:
					continue

	def dump_info(self, obj_type, fields, dir_name=None):
		"""
		writes attributes of objects in a corpus to disk.
		This function, along with load_info, supports cases where a particular attribute is to be stored separately from the other corpus files, for organization or efficiency. These attributes will not be read when the corpus is initialized; rather, they can be loaded on-demand using this function.

		For each attribute with name <NAME>, will write to a file called info.<NAME>.jsonl, where rows are json-serialized dictionaries structured as {"id": id of object, "value": value of attribute}.

		:param obj_type: type of object the attribute is associated with. can be one of "utterance", "user", "conversation".
		:param fields: a list of names of attributes to write to disk.
		:param dir_name: the directory to write attributes to. by default, or if set to None, will read from the directory that the Corpus was loaded from.
		:return: None
		"""

		if (self.original_corpus_path is None) and (dir_name is None):
			raise ValueError('must specify a directory to write to')

		if dir_name is None:
			dir_name = self.original_corpus_path
		# if len(fields) == 0:
		#     fields = self.aux_info.keys()
		for field in fields:
			# if field not in self.aux_info:
			#     raise ValueError("field %s not in index" % field)
			iterator = self.iter_objs(obj_type)
			entries = {obj.id: obj.get_info(field) for obj in iterator}
			# self._dump_jsonlist_from_dict(self.aux_info[field],
			#     os.path.join(dir_name, 'feat.%s.jsonl' % field))
			self._dump_jsonlist_from_dict(entries, os.path.join(dir_name, 'info.%s.jsonl' % field))

	def load_vector_reprs(self, field, dir_name=None):
		"""
		reads vector representations of Corpus objects from disk.

		Will read matrices from a file called vect_info.<field>.npy and corresponding object IDs from a file called vect_info.<field>.keys,

		:param field: the name of the representation
		:param dir_name: the directory to read from; by default, or if set to None, will read from the directory that the Corpus was loaded from.
		:return: None
		"""

		if (self.original_corpus_path is None) and (dir_name is None):
			raise ValueError('must specify a directory to read from')
		if dir_name is None:
			dir_name = self.original_corpus_path

		self.vector_reprs[field] = self._load_vectors(
			os.path.join(dir_name, 'vect_info.' + field)
		)

	def dump_vector_reprs(self, field, dir_name=None):
		"""
		writes vector representations of Corpus objects to disk.

		Will write matrices to a file called vect_info.<field>.npy and corresponding object IDs to a file called vect_info.<field>.keys,

		:param field: the name of the representation to write to disk
		:param dir_name: the directory to write to. by default, or if set to None, will read from the directory that the Corpus was loaded from.
		:return: None
		"""

		if (self.original_corpus_path is None) and (dir_name is None):
			raise ValueError('must specify a directory to write to')

		if dir_name is None:
			dir_name = self.original_corpus_path

		self._dump_vectors(self.vector_reprs[field], os.path.join(dir_name, 'vect_info.' + field))

	def get_attribute_table(self, obj_type, attrs):
		"""
		returns a dataframe, indexed by the IDs of objects of `obj_type`, containing attributes of these objects.

		:param obj_type: the type of object to get attributes for. can be `'utterance'`, `'user'` or `'conversation'`.
		:param attrs: a list of names of attributes to get.
		:return: a Pandas dataframe of attributes.
		"""
		iterator = self.iter_objs(obj_type)

		table_entries = []
		for obj in iterator:
			entry = dict()
			entry['id'] = obj.id
			for attr in attrs:
				entry[attr] = obj.get_info(attr)
			table_entries.append(entry)
		return pd.DataFrame(table_entries).set_index('id')

	def set_user_convo_info(self, user_id, convo_id, key, value):
		"""
		assigns user-conversation attribute `key` with `value` to user `user_id` in conversation `convo_id`.

		:param user_id: user
		:param convo_id: conversation
		:param key: name of attribute
		:param value: value of attribute
		:return: None
		"""

		user = self.get_user(user_id)
		if 'conversations' not in user.meta:
			user.meta['conversations'] = {}
		if convo_id not in user.meta['conversations']:
			user.meta['conversations'][convo_id] = {}
		user.meta['conversations'][convo_id][key] = value

	def get_user_convo_info(self, user_id, convo_id, key=None):
		"""
		retreives user-conversation attribute `key` for `user_id` in conversation `convo_id`.

		:param user_id: user
		:param convo_id: conversation
		:param key: name of attribute. if None, will return all attributes for that user-conversation.
		:return: attribute value
		"""

		user = self.get_user(user_id)
		if 'conversations' not in user.meta: return None
		if key is None:
			return user.meta['conversations'].get(convo_id, {})
		return user.meta['conversations'].get(convo_id, {}).get(key)

	def organize_user_convo_history(self, utterance_filter=None):
		"""
		For each user, pre-computes a list of all of their utterances, organized by the conversation they participated in. Annotates user with the following:
			* `n_convos`: number of conversations
			* `start_time`: time of first utterance, across all conversations
			* `conversations`: a dictionary keyed by conversation id, where entries consist of:
				* `idx`: the index of the conversation, in terms of the time of the first utterance contributed by that particular user (i.e., `idx=0` means this is the first conversation the user ever participated in)
				* `n_utterances`: the number of utterances the user contributed in the conversation
				* `start_time`: the timestamp of the user's first utterance in the conversation
				* `utterance_ids`: a list of ids of utterances contributed by the user, ordered by timestamp.
		In case timestamps are not provided with utterances, the present behavior is to sort just by utterance id.

		:param utterance_filter: function that returns True for an utterance that counts towards a user having participated in that conversation. (e.g., one could filter out conversations where the user contributed less than k words per utterance)
		"""

		if utterance_filter is None:
			utterance_filter = lambda x: True
		else:
			utterance_filter = utterance_filter

		user_to_convo_utts = defaultdict(lambda: defaultdict(list))
		for utterance in self.iter_utterances():
			if not utterance_filter(utterance): continue

			user_to_convo_utts[utterance.user.id][utterance.root].append((utterance.id, utterance.timestamp))
		for user, convo_utts in user_to_convo_utts.items():
			for convo, utts in convo_utts.items():
				sorted_utts = sorted(utts, key=lambda x: (x[1], x[0]))
				self.set_user_convo_info(user, convo, 'utterance_ids', [x[0] for x in sorted_utts])
				self.set_user_convo_info(user, convo, 'start_time', sorted_utts[0][1])
				self.set_user_convo_info(user, convo, 'n_utterances', len(sorted_utts))
		for user in self.iter_users():
			try:
				user.set_info('n_convos', len(user.get_info('conversations')))
			except:
				continue

			sorted_convos = sorted(user.get_info('conversations').items(),
								   key=lambda x: (x[1]['start_time'], x[1]['utterance_ids'][0]))
			user.set_info('start_time', sorted_convos[0][1]['start_time'])
			for idx, (convo_id, _) in enumerate(sorted_convos):
				self.set_user_convo_info(user.id, convo_id, 'idx', idx)

	def get_user_convo_attribute_table(self, attrs):
		"""
		returns a table where each row lists a (user, convo) level aggregate for each attribute in attrs.

		:param attrs: list of (user, convo) attribute names
		:return: dataframe containing all user,convo attributes.
		"""

		table_entries = []
		for user in self.iter_users():

			if 'conversations' not in user.meta: continue
			for convo_id, convo_dict in user.meta['conversations'].items():
				entry = {'id': '%s__%s' % (user.id, convo_id),
						 'user': user.id, 'convo_id': convo_id,
						 'convo_idx': convo_dict['idx']}

				for attr in attrs:
					entry[attr] = convo_dict.get(attr, None)
				table_entries.append(entry)
		return pd.DataFrame(table_entries).set_index('id')

	def get_full_attribute_table(self, user_convo_attrs, user_attrs=None, convo_attrs=None, user_suffix='__user',
								 convo_suffix='__convo'):
		"""
		returns a table where each row lists a (user, convo) level aggregate for each attribute in attrs, along with user-level and conversation-level attributes; by default these attributes are suffixed with '__user' and '__convo' respectively.

		:param user_convo_attrs: list of (user, convo) attribute names
		:param user_attrs: list of user attribute names
		:param convo_attrs: list of conversation attribute names
		:param user_suffix: suffix to append to names of user-level attributes
		:param convo_suffix: suffix to append to names of conversation-level attributes.
		:return: dataframe containing all attributes.
		"""
		if user_attrs is None:
			user_attrs = []
		if convo_attrs is None:
			convo_attrs = []

		uc_df = self.get_user_convo_attribute_table(user_convo_attrs)
		u_df = self.get_attribute_table('user', user_attrs)
		u_df.columns = [x + user_suffix for x in u_df.columns]
		c_df = self.get_attribute_table('conversation', convo_attrs)
		c_df.columns = [x + convo_suffix for x in c_df.columns]
		return uc_df.join(u_df, on='user').join(c_df, on='convo_id')

# def __repr__(self):
# def __eq__(self, other):
# 	return True
