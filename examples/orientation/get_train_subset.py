import os
import pandas as pd
from multiprocessing import Pool

from convokit import Corpus, download
from convokit.text_processing import TextProcessor, TextToArcs
from convokit.convokitPipeline import ConvokitPipeline

import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Pool

DATA_DIR = '/kitchen/experimental_justine/scotus/demo_data'
TO_DOWNLOAD = True

MIN_YEAR = 1955
MAX_YEAR = 2019

N_JOBS = 16

TEXT_COLS = ['arcs','tokens']
MIN_WC_SOURCE = 10
MAX_WC_SOURCE = 50
MIN_WC_TARGET = 10
MAX_WC_TARGET = 75
SOURCE_FILTER = lambda utt: (utt.retrieve_meta('speaker_type') == 'J') and (utt.retrieve_meta('arcs') != '')
TARGET_FILTER = lambda utt: (utt.retrieve_meta('speaker_type') == 'A') and (utt.retrieve_meta('arcs') != '')



def get_context_id_df(corpus):
	prev_df = pd.DataFrame([{'id': utt.id, 'prev_id': utt.reply_to} for utt in corpus.iter_utterances()])
	context_id_df = prev_df.join(prev_df.drop_duplicates('prev_id').set_index('prev_id')['id'].rename('next_id'), on='id')
	return context_id_df

def text_prep_pipe():
	return ConvokitPipeline([
		('arcs_per_sent', TextToArcs(output_field='arcs_per_sent')),
		('arcs', TextProcessor(input_field='arcs_per_sent', output_field='arcs',
						 proc_fn=lambda sents: '\n'.join(sents))),
		('wordcount', TextProcessor(input_field='parsed', output_field='wordcount',
			   proc_fn=lambda sents: sum(sum(x['tag'] != '_SP' for x in sent['toks']) for sent in sents))),
		('tokens', TextProcessor(input_field='parsed', output_field='tokens',
			   proc_fn=lambda sents: '\n'.join((' '.join(x['tok'] for x in sent['toks']).strip()) for sent in sents)))
	])

def get_train_subset(corpus, 
		min_wc_source, max_wc_source,
		min_wc_target, max_wc_target,
		source_filter, target_filter, text_cols):

	context_id_df = get_context_id_df(corpus)

	for utt in corpus.iter_utterances():
		utt.set_info('source_filter',source_filter(utt))
		utt.set_info('target_filter',target_filter(utt))
	utt_df = corpus.get_attribute_table('utterance', ['wordcount', 'source_filter','target_filter'])

	full_context_df = context_id_df.join(utt_df, on='id')\
		.join(utt_df, on='prev_id', rsuffix='_prev')\
		.join(utt_df, on='next_id', rsuffix='_next')

	source_df = full_context_df[full_context_df.source_filter
           & full_context_df.wordcount.between(min_wc_source, max_wc_source)
           & full_context_df.wordcount_prev.between(min_wc_target, max_wc_target)
           & full_context_df.wordcount_next.between(min_wc_target, max_wc_target)].set_index('id')
	target_df = full_context_df[full_context_df.target_filter
   		& full_context_df.wordcount.between(min_wc_target, max_wc_target)].set_index('id')

	source_df = source_df[source_df.prev_id.isin(target_df.index)
         & source_df.next_id.isin(target_df.index)]

	text_df = corpus.get_attribute_table('utterance',text_cols)	 

	source_df = source_df[['prev_id','next_id']].join(text_df)
	target_df = target_df[[]].join(text_df)
	return source_df, target_df

def process_corpus(corpus_name, to_download=TO_DOWNLOAD,
		min_wc_source=MIN_WC_SOURCE, max_wc_source=MAX_WC_SOURCE,
		min_wc_target=MIN_WC_TARGET, max_wc_target=MAX_WC_TARGET,
		source_filter=SOURCE_FILTER, target_filter=TARGET_FILTER,
		text_cols=TEXT_COLS, data_dir=DATA_DIR):
	
	if to_download:
		corpus = Corpus(download(corpus_name, data_dir=data_dir))
	else:
		corpus = Corpus(os.path.join(data_dir, corpus_name))
	corpus_name = corpus.get_meta()['name']
	print(corpus_name)
	corpus.print_summary_stats()
	print('processing', corpus.get_meta()['name'])
	corpus.load_info('utterance',['parsed'])

	corpus = text_prep_pipe().transform(corpus)

	source_df, target_df = get_train_subset(corpus, 
		min_wc_source, max_wc_source,
		min_wc_target, max_wc_target,
		source_filter, target_filter, text_cols)
	source_df.to_csv(os.path.join(data_dir, corpus_name + '.source.tsv'), sep='\t')
	target_df.to_csv(os.path.join(data_dir, corpus_name + '.target.tsv'), sep='\t')

if __name__ == '__main__':
	corpus_names = ['oyez_%s' % year for year in range(MIN_YEAR, MAX_YEAR + 1)]
	pool = Pool(N_JOBS)
	pool.map(process_corpus, corpus_names)