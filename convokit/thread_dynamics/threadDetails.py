from convokit import Transformer, Corpus

class ThreadDetails(Transformer):
    """
    Annotates Utterances with their Conversational depth, with the root Utterance having a depth of 1
    """
    def transform(self, corpus: Corpus) -> Corpus:
        for convo in corpus.iter_conversations():
            for utt in convo.traverse('bfs'):
                if utt.reply_to is None:
                    utt.meta['depth'] = 0
                else:
                    utt.meta['depth'] = convo.get_utterance(utt.reply_to).meta['depth'] + 1

            for idx, utt in enumerate(convo.get_chronological_utterance_list()):
                utt.meta['chrono_order'] = idx + 1

        return corpus
