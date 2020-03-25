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

CORPUS_DIR = "longreddit_construction/long-reddit-corpus"
LIWC_CORPUS_DIR = "longreddit_construction/long-reddit-corpus-liwc"
# CORPUS_DIR = "reddit-corpus-small"
# CORPUS_DIR =
DATA_DIR = "data_liwc"
PLOT_DIR = "html/graphs_liwc"
# hyperconv_range = range(0, 9+1)
hyperconv_range = range(3, 20+1)
rank_range = range(9, 9+1)
max_rank = max(rank_range)
anomaly_threshold = 1.5

def save_corpus_details(corpus):
    subreddits = [convo.get_utterance(convo.id).meta['subreddit'] for convo in corpus.iter_conversations()]
    convo_ids = [convo.id for convo in corpus.iter_conversations()]

    with open(os.path.join(DATA_DIR, 'subreddits.p'), 'wb') as f:
        pickle.dump(subreddits, f)

    with open(os.path.join(DATA_DIR, 'convo_ids.p'), 'wb') as f:
        pickle.dump(convo_ids, f)
    return subreddits, convo_ids

def multi_hyperconv_transform(corpus, hyperconv_range):
    hc_transformers = [HyperConvo(prefix_len=i, feat_name="hyperconvo-{}".format(i)) for i in hyperconv_range]
    for idx, hc in enumerate(list(reversed(hc_transformers))):
        print(hyperconv_range[-1]-idx)
        hc.transform(corpus)

def construct_tensor(corpus, hyperconv_range, impute_na=None):
    """

    :param corpus:
    :param hyperconv_range:
    :param impute_na: If set to an int, replace all NaNs with the set integer.
    :return:
    """
    num_convos = len(list(corpus.iter_conversations()))
    num_feature_sets = len(hyperconv_range)
    tensor = np.zeros((num_feature_sets, num_convos, 140))

    for convo_idx, convo in enumerate(corpus.iter_conversations()):
        for hyperconvo_idx in hyperconv_range:
            tensor[hyperconvo_idx-3][convo_idx] = list(convo.meta['hyperconvo-{}'.format(hyperconvo_idx)].values())

    if impute_na is not None:
        tensor[np.isnan(tensor)] = impute_na
    return tensor

def generate_data_and_tensor(sliding=False):
    print("Loading corpus from {}...".format(CORPUS_DIR), end="")
    corpus = Corpus(filename=CORPUS_DIR)
    print("Done.\n")

    # getting corpus details
    subreddits, convo_ids = save_corpus_details(corpus)

    print("Doing Hyperconvo Transformations...")
    if sliding:
        hyperconv = HyperConvo(prefix_len=10, min_thread_len=20)
        hyperconv.sliding_transform(corpus)
    else:
        multi_hyperconv_transform(corpus, hyperconv_range)
    print("Done.\n")

    print("Constructing tensor...", end="")
    tensor = construct_tensor(corpus, hyperconv_range, impute_na=-1)
    print("Done.\n")

    print("Saving tensor...", end="")
    with open(os.path.join(DATA_DIR, 'tensor.p'), 'wb') as f:
        pickle.dump(tensor, f)
    hg_features = list(next(corpus.iter_conversations()).meta['hyperconvo-3'])
    with open(os.path.join(DATA_DIR, 'hg_features.p'), 'wb') as f:
        pickle.dump(hg_features, f)
    print("Saved.\n")

def generate_liwc_data_and_tensor(sliding=False):
    print("Loading corpus from {}...".format(LIWC_CORPUS_DIR), end="")
    corpus = Corpus(filename=LIWC_CORPUS_DIR)
    print("Done.\n")

    # getting corpus details
    subreddits, convo_ids = save_corpus_details(corpus)

    print("Constructing tensor...", end="")

    num_convos = len(list(corpus.iter_conversations()))
    num_feature_sets = len(corpus.random_utterance())
    tensor = np.zeros((num_feature_sets, num_convos, 140))

    for convo_idx, convo in enumerate(corpus.iter_conversations()):
        for hyperconvo_idx in hyperconv_range:
            tensor[hyperconvo_idx-3][convo_idx] = list(convo.meta['hyperconvo-{}'.format(hyperconvo_idx)].values())

    if impute_na is not None:
        tensor[np.isnan(tensor)] = impute_na

    tensor = construct_tensor(corpus, hyperconv_range, impute_na=-1)
    print("Done.\n")

    print("Saving tensor...", end="")
    with open(os.path.join(DATA_DIR, 'tensor.p'), 'wb') as f:
        pickle.dump(tensor, f)
    hg_features = list(next(corpus.iter_conversations()).meta['hyperconvo-3'])
    with open(os.path.join(DATA_DIR, 'hg_features.p'), 'wb') as f:
        pickle.dump(hg_features, f)
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

    with open(os.path.join(DATA_DIR, 'hg_features.p'), 'rb') as f:
        hg_features = pickle.load(f)

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
        idx_to_distinctive_features[idx]['pos_features'] = [hg_features[i] for i in pos_features]
        idx_to_distinctive_features[idx]['neg_features'] = [hg_features[i] for i in neg_features]

    factor_to_details = dict()
    for idx in range(max(rank_range)):
        factor_to_details[idx] = dict()
        pos_subreddits = sorted(list(idx_to_distinctive_threads[idx]['pos_threads'].items()),
                                key=lambda x: x[1], reverse=True)
        factor_to_details[idx]['pos_subreddits'] = [k for k, v in pos_subreddits[:5]]
        neg_subreddits = sorted(list(idx_to_distinctive_threads[idx]['neg_threads'].items()),
                                key=lambda x: x[1], reverse=True)
        factor_to_details[idx]['neg_subreddits'] = [k for k, v in neg_subreddits[:5]]

        factor_to_details[idx]['pos_features'] = get_graphic_dict(idx_to_distinctive_features[idx]['pos_features'][:10])
        factor_to_details[idx]['neg_features'] = get_graphic_dict(idx_to_distinctive_features[idx]['neg_features'][:10])

    return factor_to_details

def get_convo_details(convo):
    print("Subreddit: {}".format(convo.get_utterance(convo.id).meta['subreddit']))
    convo.print_conversation_structure(lambda utt: str(utt.meta['order']) + ". " + utt.user.id)

def generate_detailed_examples():
    print("Generating detailed examples for manual examination")
    with open(os.path.join(DATA_DIR, 'rank_to_factors.p'), 'rb') as f:
        rank_to_factors = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'hg_features.p'), 'rb') as f:
        hg_features = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'subreddits.p'), 'rb') as f:
        subreddits = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'convo_ids.p'), 'rb') as f:
        convo_ids = pickle.load(f)

    time_factor = rank_to_factors[max_rank][0] # (9, 9)
    thread_factor = rank_to_factors[max_rank][1] # (10000, 9)
    feature_factor = rank_to_factors[max_rank][2] # (140, 9)

    print("Reloading corpus...", end="")
    corpus = Corpus(filename=CORPUS_DIR)
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
    template = env.get_template('report.html')
    filename = os.path.join(root, 'html', output_html)
    with open(filename, 'w') as fh:
        fh.write(
            template.render(title=title, factor_to_details=factor_to_details, graph_filepath=graph_filepath)
        )
    # pdf = HTML('html/report.html').write_pdf('html/report.pdf')



if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    generate_data_and_tensor(sliding=False)
    decompose_tensor()
    generate_plots()
    generate_html(generate_high_level_summary(),
                  title="Report - LIWC",
                  graph_filepath='graphs_liwc',
                  output_html='report_sliding_fixed.html')

    # generate_detailed_examples()













