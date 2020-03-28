"""
Run only on fully intact corpora.
Conversations ARE threads.
"""
from convokit import Corpus, HyperConvo
import pickle
import numpy as np
import os
from tensorly.decomposition import parafac
from convokit.tensors.utils import plot_factors
from convokit.tensors.graphics import get_graphic_dict
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings

IMAGE_WIDTH = 50
warnings.filterwarnings('error')

LIWC_CORPUS_DIR = "longreddit_construction/long-reddit-corpus-liwc"
# CORPUS_DIR = "reddit-corpus-small"
# CORPUS_DIR =
DATA_DIR = "data_liwc"
PLOT_DIR = "html/graphs_liwc"
# hyperconv_range = range(0, 9+1)
rank_range = range(9, 9+1)
max_rank = max(rank_range)
anomaly_threshold = 1.5
WINDOW_SIZE = 10

def save_corpus_details(corpus):
    subreddits = [convo.get_utterance(convo.id).meta['subreddit'] for convo in corpus.iter_conversations()]
    convo_ids = [convo.id for convo in corpus.iter_conversations()]

    with open(os.path.join(DATA_DIR, 'subreddits.p'), 'wb') as f:
        pickle.dump(subreddits, f)

    with open(os.path.join(DATA_DIR, 'convo_ids.p'), 'wb') as f:
        pickle.dump(convo_ids, f)
    return subreddits, convo_ids

def generate_liwc_data_and_tensor(sliding=False):
    print("Loading corpus from {}...".format(LIWC_CORPUS_DIR), end="")
    corpus = Corpus(filename=LIWC_CORPUS_DIR)
    print("Done.\n")

    # getting corpus details
    subreddits, convo_ids = save_corpus_details(corpus)

    print("Constructing tensor...", end="")

    num_convos = len(list(corpus.iter_conversations()))
    # sliding window of size 10, so we get 11 windows
    num_feature_sets = 11
    num_liwc_feats = len(corpus.random_conversation().get_chronological_utterance_list()[0].meta['liwc'])
    tensor = np.zeros((num_feature_sets, num_convos, num_liwc_feats))

    for convo_idx, convo in enumerate(corpus.iter_conversations()):
        liwc_mat = convo.meta['liwc']
        for idx in range(0, 20+1-WINDOW_SIZE):
            tensor[idx][convo_idx] = liwc_mat[idx:idx+WINDOW_SIZE].sum(axis=0)

    print("Done.\n")

    print("Saving tensor...", end="")
    with open(os.path.join(DATA_DIR, 'tensor.p'), 'wb') as f:
        pickle.dump(tensor, f)

    liwc_features = list(corpus.random_conversation().get_chronological_utterance_list()[0].meta['liwc'])
    with open(os.path.join(DATA_DIR, 'liwc_features.p'), 'wb') as f:
        pickle.dump(liwc_features, f)
    print("Saved.\n")

def decompose_tensor():
    print("Decomposing tensor into factors...")
    with open(os.path.join(DATA_DIR, 'tensor.p'), 'rb') as f:
        tensor = pickle.load(f)
    rank_to_factors = dict()
    for rank in rank_range:
        print("Rank {}".format(rank))
        rank_to_factors[rank] = parafac(tensor, rank=rank)[1]
    with open(os.path.join(DATA_DIR, 'rank_to_factors.p'), 'wb') as f:
        pickle.dump(rank_to_factors, f)
    print("Finished decomposition and saved.\n")

def plot_factors(factors, d=3):
    a, b, c = factors
    rank = a.shape[1]
    for factor_idx in range(rank):
        fig, ax = plt.subplots(1, d, figsize=(8, int(1.2+1)))
        ax[0].set_ylabel("Factor " + str(factor_idx+1))
        factors_name = ["Time", "Threads", "Features"] if d==3 else ["Time", "Features"]
        for col_idx in range(d):
            sns.despine(top=True, ax=ax[col_idx])
            ax[col_idx].plot(factors[col_idx].T[factor_idx])
            ax[col_idx].set_xlabel(factors_name[col_idx])
    # ax[1].set_xlabel("hey")
        plt.savefig(os.path.join(PLOT_DIR, 'factorplot_{}.png'.format(factor_idx+1)))
    # fig, axes = plt.subplots(rank, d, figsize=(8, int(rank * 1.2 + 1)))
    # for idx,
    # for ind, (factor, axs) in enumerate(zip(factors[:d], axes.T)):
    #     axs[-1].set_xlabel(factors_name[ind])
    #     for i, (f, ax) in enumerate(zip(factor.T, axs)):
    #         sns.despine(top=True, ax=ax)
    #         ax.plot(f)
    #         axes[i, 0].set_ylabel("Factor " + str(i+1))
    # fig.tight_layout()
    # plt.savefig(os.path.join(PLOT_DIR, 'tester.png'))

def generate_plots():
    print("Generating plots...")
    with open(os.path.join(DATA_DIR, 'rank_to_factors.p'), 'rb') as f:
        rank_to_factors = pickle.load(f)

    plot_factors(rank_to_factors[max_rank])
    plt.show()


def get_anomalous_points(factor_full, idx):
    scaler = StandardScaler()
    factor = factor_full[:, idx]
    reshaped = factor.reshape((factor.shape[0], 1))
    scaled = scaler.fit_transform(reshaped)
    pos_pts = np.argwhere(scaled.reshape(factor.shape[0]) > anomaly_threshold).flatten()
    neg_pts = np.argwhere(scaled.reshape(factor.shape[0]) < -anomaly_threshold).flatten()
    return pos_pts, neg_pts


def generate_high_level_summary():
    # generate_plots()
    with open(os.path.join(DATA_DIR, 'rank_to_factors.p'), 'rb') as f:
        rank_to_factors = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'liwc_features.p'), 'rb') as f:
        liwc_features = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'subreddits.p'), 'rb') as f:
        subreddits = pickle.load(f)

    time_factor = rank_to_factors[max_rank][0] # (9, 9)
    thread_factor = rank_to_factors[max_rank][1] # (10000, 9)
    feature_factor = rank_to_factors[max_rank][2] # (140, 9)
    idx_to_distinctive_threads = defaultdict(dict)
    idx_to_distinctive_features = defaultdict(dict)

    # normalizing
    subreddit_totals = Counter(subreddits)
    for idx in range(max_rank):
        pos_thread_pts, neg_thread_pts = get_anomalous_points(thread_factor, idx)
        idx_to_distinctive_threads[idx]['pos_threads'] = Counter([subreddits[i] for i in pos_thread_pts])
        idx_to_distinctive_threads[idx]['neg_threads'] = Counter([subreddits[i] for i in neg_thread_pts])

        # normalize subreddit counts
        for subreddit in idx_to_distinctive_threads[idx]['pos_threads']:
            idx_to_distinctive_threads[idx]['pos_threads'][subreddit] /= subreddit_totals[subreddit]
            idx_to_distinctive_threads[idx]['neg_threads'][subreddit] /= subreddit_totals[subreddit]

        pos_features, neg_features = get_anomalous_points(feature_factor, idx)
        idx_to_distinctive_features[idx]['pos_features'] = [liwc_features[i] for i in pos_features]
        idx_to_distinctive_features[idx]['neg_features'] = [liwc_features[i] for i in neg_features]

    factor_to_details = dict()
    for idx in range(max(rank_range)):
        factor_to_details[idx] = dict()
        pos_subreddits = sorted(list(idx_to_distinctive_threads[idx]['pos_threads'].items()),
                                key=lambda x: x[1], reverse=True)
        factor_to_details[idx]['pos_subreddits'] = [k for k, v in pos_subreddits[:5]]
        neg_subreddits = sorted(list(idx_to_distinctive_threads[idx]['neg_threads'].items()),
                                key=lambda x: x[1], reverse=True)
        factor_to_details[idx]['neg_subreddits'] = [k for k, v in neg_subreddits[:5]]

        factor_to_details[idx]['pos_features'] = ', '.join(idx_to_distinctive_features[idx]['pos_features'][:10])
        factor_to_details[idx]['neg_features'] = ', '.join(idx_to_distinctive_features[idx]['neg_features'][:10])

    return factor_to_details

def get_convo_details(convo):
    print("Subreddit: {}".format(convo.get_utterance(convo.id).meta['subreddit']))
    convo.print_conversation_structure(lambda utt: str(utt.meta['order']) + ". " + utt.user.id)

def generate_detailed_examples():
    print("Generating detailed examples for manual examination")
    with open(os.path.join(DATA_DIR, 'rank_to_factors.p'), 'rb') as f:
        rank_to_factors = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'liwc_features.p'), 'rb') as f:
        liwc_features = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'subreddits.p'), 'rb') as f:
        subreddits = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'convo_ids.p'), 'rb') as f:
        convo_ids = pickle.load(f)

    time_factor = rank_to_factors[max_rank][0] # (9, 9)
    thread_factor = rank_to_factors[max_rank][1] # (10000, 9)
    feature_factor = rank_to_factors[max_rank][2] # (140, 9)

    print("Reloading corpus...", end="")
    corpus = Corpus(filename=LIWC_CORPUS_DIR)
    print("Done.\n")

    print("Annotating utterances with arrival information...", end="")
    for convo in corpus.iter_conversations():
        for idx, utt in enumerate(convo.get_chronological_utterance_list()):
            utt.meta['order'] = idx
    print("Done.\n")

    convos = list(corpus.iter_conversations())
    for idx in range(max(rank_range)):
        print("### FACTOR {} ###".format(idx+1))
        pos_threads, neg_threads = get_anomalous_points(thread_factor, idx)

        print("Positive thread examples\n")
        for idx in random.sample(pos_threads, 5):
            get_convo_details(convos[idx])
            print()

        print("Negative thread examples\n")
        for idx in random.sample(neg_threads, 5):
            get_convo_details(convos[idx])
            print()

        print("#########################################\n\n")


def generate_html(factor_to_details, title="Report", graph_filepath='graphs', output_html='report.html'):
    root = os.path.dirname(os.path.abspath(__file__))
    factor_to_details = generate_high_level_summary()
    env = Environment(loader=FileSystemLoader(os.path.join(root, 'template')))
    template = env.get_template('liwc_report.html')
    filename = os.path.join(root, 'html', output_html)
    with open(filename, 'w') as fh:
        fh.write(
            template.render(title=title, factor_to_details=factor_to_details, graph_filepath=graph_filepath)
        )
    # pdf = HTML('html/report.html').write_pdf('html/report.pdf')



if __name__ == "__main__":
    # os.makedirs(DATA_DIR, exist_ok=True)
    # os.makedirs(PLOT_DIR, exist_ok=True)
    # generate_liwc_data_and_tensor()
    # decompose_tensor()
    # generate_plots()
    generate_html(generate_high_level_summary(),
                  title="Report - LIWC",
                  graph_filepath='graphs_liwc',
                  output_html='report_liwc.html')

    # generate_detailed_examples()













