from typing import Tuple, List, Dict, Collection
from collections import defaultdict
from convokit import Utterance, User

class Hypergraph:
    """
    Represents a hypergraph, consisting of nodes, directed edges,
    hypernodes (each of which is a set of nodes) and hyperedges (directed edges
    from hypernodes to hypernodes). Contains functionality to extract motifs
    from hypergraphs (Fig 2 of
    http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html)
    """
    def __init__(self):
        # public
        self.nodes: Dict[str, Utterance] = dict()
        self.hypernodes = dict()
        self.users = dict()

        # private
        self.adj_out = dict()  # out edges for each (hyper)node
        self.adj_in = dict()   # in edges for each (hyper)node

    @staticmethod
    def init_from_utterances(utterances: List[Utterance]):
        utt_dict = {utt.id: utt for utt in utterances}
        utt_to_user_id = {utt.id: utt.user.id for utt in utterances}
        hypergraph = Hypergraph()
        user_to_utt_ids = dict()
        reply_edges = []
        speaker_to_reply_tos = defaultdict(list)
        speaker_target_pairs = list()

        # nodes (utts)
        for utt in sorted(utterances, key=lambda h: h.timestamp):
            if utt.user not in user_to_utt_ids:
                user_to_utt_ids[utt.user] = set()
            user_to_utt_ids[utt.user].add(utt.id)

            if utt.reply_to is not None and utt.reply_to in utt_dict:
                reply_edges.append((utt.id, utt.reply_to))
                speaker_to_reply_tos[utt.user.id].append(utt.reply_to)
                speaker_target_pairs.append([utt.user.id, utt_dict[utt.reply_to].user.id,
                                          {'utt': utt, 'target_user': utt_to_user_id[utt.reply_to]}])
            hypergraph.add_node(utt)

        # hypernodes (users)
        for user, utt_ids in user_to_utt_ids.items():
            hypergraph.add_hypernode(user, utt_ids)

        # reply edges (utt to utt)
        for speaker_utt_id, target_utt_id in reply_edges:
            hypergraph.add_edge(speaker_utt_id, target_utt_id)

        # user to utterance response edges
        for user, reply_tos in speaker_to_reply_tos.items():
            for reply_to in reply_tos:
                hypergraph.add_edge(user, reply_to)

        # user to user response edges
        for user, target, utt_info in speaker_target_pairs:
            hypergraph.add_edge(user, target, utt_info)

        return hypergraph

    def add_node(self, utt: Utterance) -> None:
        self.nodes[utt.id] = utt
        self.adj_out[utt.id] = dict()
        self.adj_in[utt.id] = dict()

    def add_hypernode(self, user: User, nodes: Collection[str]) -> None:
        self.hypernodes[user.id] = set(nodes)
        self.users[user.id] = user
        self.adj_out[user.id] = dict()
        self.adj_in[user.id] = dict()

    # edge or hyperedge
    def add_edge(self, u: str, v: str, info=None) -> None:
        assert u in self.nodes or u in self.hypernodes
        assert v in self.nodes or v in self.hypernodes
        # if u in self.hypernodes and v in self.hypernodes:
        #     assert info is not N
        if v not in self.adj_out[u]:
            self.adj_out[u][v] = []
        if u not in self.adj_in[v]:
            self.adj_in[v][u] = []
        if info is None: info = dict()
        self.adj_out[u][v].append(info)
        self.adj_in[v][u].append(info)

    def edges(self) -> Dict[Tuple[str, str], List]:
        return dict(((u, v), lst) for u, d in self.adj_out.items()
                    for v, lst in d.items())

    def outgoing_nodes(self, u: str) -> Dict[str, List]:
        assert u in self.adj_out
        return dict((v, lst) for v, lst in self.adj_out[u].items()
                    if v in self.nodes)

    def outgoing_hypernodes(self, u) -> Dict[str, List]:
        assert u in self.adj_out
        return dict((v, lst) for v, lst in self.adj_out[u].items()
                    if v in self.hypernodes)

    def incoming_nodes(self, v: str) -> Dict[str, List]:
        assert v in self.adj_in
        return dict((u, lst) for u, lst in self.adj_in[v].items() if u in
                    self.nodes)

    def incoming_hypernodes(self, v: str) -> Dict[str, List]:
        assert v in self.adj_in
        return dict((u, lst) for u, lst in self.adj_in[v].items() if u in
                    self.hypernodes)

    def outdegrees(self, from_hyper: bool=False, to_hyper: bool=False) -> List[int]:
        return [sum([len(l) for v, l in self.adj_out[u].items() if v in
                     (self.hypernodes if to_hyper else self.nodes)]) for u in
                (self.hypernodes if from_hyper else self.nodes)]

    def indegrees(self, from_hyper: bool=False, to_hyper: bool=False) -> List[int]:
        return [sum([len(l) for u, l in self.adj_in[v].items() if u in
                     (self.hypernodes if from_hyper else self.nodes)]) for v in
                (self.hypernodes if to_hyper else self.nodes)]

